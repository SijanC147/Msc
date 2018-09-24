import tensorflow as tf
from tensorflow.estimator import (  # pylint: disable=E0401
    EstimatorSpec,
    ModeKeys,
)
from tensorflow.contrib.rnn import (  # pylint: disable=E0611
    stack_bidirectional_dynamic_rnn
)
from tsaplay.models.Model import Model
from tsaplay.models.Chen2017.common import (
    params as default_params,
    ram_input_fn,
    ram_serving_fn,
    get_bounded_distance_vectors,
    calculate_u_t_w_t,
    get_location_weighted_memory,
    var_len_concatenate,
    ram_attn_unit,
)
from tsaplay.utils._tf import (
    variable_len_batch_mean,
    attention_unit,
    dropout_lstm_cell,
    dropout_gru_cell,
    l2_regularized_loss,
    generate_attn_heatmap_summary,
    setup_embedding_layer,
    get_embedded_seq,
    create_snapshots_container,
    append_snapshot,
    zip_attn_snapshots_with_literals,
)


class RecurrentAttentionNetwork(Model):
    def _params(self):
        return default_params

    def _feature_columns(self):
        default = []
        return default

    def _train_input_fn(self):
        return lambda features, labels, batch_size: ram_input_fn(
            features, labels, batch_size
        )

    def _eval_input_fn(self):
        return lambda features, labels, batch_size: ram_input_fn(
            features, labels, batch_size, eval_input=True
        )

    def _serving_input_fn(self):
        return lambda features: ram_serving_fn(features)

    def _model_fn(self):
        def default(features, labels, mode, params=self.params):
            embedding_matrix = setup_embedding_layer(
                vocab_size=params["vocab_size"],
                dim_size=params["embedding_dim"],
                init=params["embedding_initializer"],
            )

            sentence_embeddings = get_embedded_seq(
                features["sentence_x"], embedding_matrix
            )
            target_embeddings = get_embedded_seq(
                features["target_x"], embedding_matrix
            )

            batch_size = tf.shape(sentence_embeddings)[0]
            max_seq_len = tf.shape(sentence_embeddings)[1]

            forward_cells = []
            backward_cells = []
            for n in range(params["n_lstm_layers"]):
                forward_cells.append(
                    dropout_lstm_cell(
                        hidden_units=params["lstm_hidden_units"],
                        initializer=params["initializer"],
                        keep_prob=params["keep_prob"],
                    )
                )
                backward_cells.append(
                    dropout_lstm_cell(
                        hidden_units=params["lstm_hidden_units"],
                        initializer=params["initializer"],
                        keep_prob=params["keep_prob"],
                    )
                )

            with tf.variable_scope("bi_lstm"):
                memory_star, _, _ = stack_bidirectional_dynamic_rnn(
                    cells_fw=forward_cells,
                    cells_bw=backward_cells,
                    inputs=sentence_embeddings,
                    sequence_length=features["sentence_len"],
                    dtype=tf.float32,
                )

            distances = get_bounded_distance_vectors(
                left_bounds=features["target_left_bound"],
                right_bounds=features["target_right_bound"],
                seq_lens=features["sentence_len"],
                max_seq_len=max_seq_len,
            )

            u_t, w_t = calculate_u_t_w_t(
                features["sentence_len"], max_seq_len, distances
            )

            memory = get_location_weighted_memory(memory_star, w_t, u_t)

            episode_0 = tf.zeros(
                shape=[batch_size, params["gru_hidden_units"], 1]
            )

            target_avg = variable_len_batch_mean(
                input_tensor=target_embeddings,
                seq_lengths=features["target_len"],
                op_name="target_avg_pooling",
            )

            attn_snapshots = create_snapshots_container(
                shape_like=features["sentence_x"],
                n_snaps=params["n_attn_layers"],
            )

            attn_layer_num = tf.constant(1)

            initial_layer_inputs = (attn_layer_num, episode_0, attn_snapshots)

            def condition(attn_layer_num, episode, attn_snapshots):
                return tf.less_equal(attn_layer_num, params["n_attn_layers"])

            def attn_layer_run(attn_layer_num, episode, attn_snapshots):
                mem_prev_ep_v_target = var_len_concatenate(
                    seq_lens=features["sentence_len"],
                    memory=memory,
                    v_target=target_avg,
                    prev_episode=episode,
                )

                weight_dim = (
                    (params["lstm_hidden_units"] * 2)
                    + 1
                    + params["gru_hidden_units"]
                    + params["embedding_dim"]
                )

                with tf.variable_scope("attention_layer", reuse=tf.AUTO_REUSE):
                    attn_scores = ram_attn_unit(
                        seq_lens=features["sentence_len"],
                        attn_focus=mem_prev_ep_v_target,
                        weight_dim=weight_dim,
                        init=params["initializer"],
                    )

                content_i_al = tf.reduce_sum(
                    memory * attn_scores, axis=1, keepdims=True
                )

                with tf.variable_scope("gru_layer", reuse=tf.AUTO_REUSE):
                    _, final_state = tf.nn.dynamic_rnn(
                        cell=dropout_gru_cell(
                            hidden_units=params["gru_hidden_units"],
                            initializer=params["initializer"],
                            keep_prob=params["keep_prob"],
                        ),
                        inputs=content_i_al,
                        sequence_length=features["sentence_len"],
                        dtype=tf.float32,
                    )

                final_state = tf.expand_dims(final_state, axis=1)

                attn_snapshots = append_snapshot(
                    container=attn_snapshots,
                    new_snap=attn_scores,
                    index=attn_layer_num,
                )

                attn_layer_num = tf.add(attn_layer_num, 1)

                return (attn_layer_num, final_state, attn_snapshots)

            _, final_episode, attn_snapshots = tf.while_loop(
                cond=condition,
                body=attn_layer_run,
                loop_vars=initial_layer_inputs,
            )

            literals, attn_snapshots = zip_attn_snapshots_with_literals(
                literals=features["sentence_lit"],
                snapshots=attn_snapshots,
                num_layers=params["n_attn_layers"],
            )
            attn_info = tf.tuple([literals, attn_snapshots])
            generate_attn_heatmap_summary(attn_info)

            final_sentence_rep = tf.squeeze(final_episode, axis=1)

            logits = tf.layers.dense(
                inputs=final_sentence_rep, units=params["n_out_classes"]
            )

            predictions = {
                "class_ids": tf.argmax(logits, 1),
                "probabilities": tf.nn.softmax(logits),
                "logits": logits,
            }

            if mode == ModeKeys.PREDICT:
                return EstimatorSpec(mode, predictions=predictions)

            loss = l2_regularized_loss(
                labels=labels, logits=logits, l2_weight=params["l2_weight"]
            )

            if mode == ModeKeys.EVAL:
                return EstimatorSpec(mode, predictions=predictions, loss=loss)

            optimizer = tf.train.GradientDescentOptimizer(
                learning_rate=params["learning_rate"]
            )

            train_op = optimizer.minimize(
                loss, global_step=tf.train.get_global_step()
            )

            return EstimatorSpec(
                mode, loss=loss, train_op=train_op, predictions=predictions
            )

        return default