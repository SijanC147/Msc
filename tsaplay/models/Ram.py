import tensorflow as tf
from tensorflow.estimator import (  # pylint: disable=E0401
    EstimatorSpec,
    ModeKeys,
)
from tensorflow.contrib.rnn import (  # pylint: disable=E0611
    stack_bidirectional_dynamic_rnn
)
from tsaplay.models.TSAModel import TSAModel
from tsaplay.models.addons import attn_heatmaps
from tsaplay.utils.decorators import addon
from tsaplay.utils.tf import (
    masked_softmax,
    sparse_sequences_to_dense,
    seq_lengths,
    variable_len_batch_mean,
    dropout_lstm_cell,
    dropout_gru_cell,
    l2_regularized_loss,
    generate_attn_heatmap_summary,
    setup_embedding_layer,
    get_embedded_seq,
    create_snapshots_container,
    append_snapshot,
    zip_attn_snapshots_with_sp_literals,
)


class Ram(TSAModel):
    def set_params(self):
        return {
            "batch_size": 25,
            "n_out_classes": 3,
            "learning_rate": 0.1,
            "l2_weight": 1e-5,
            "keep_prob": 0.5,
            "lstm_hidden_units": 100,
            "gru_hidden_units": 50,
            "n_lstm_layers": 2,
            "n_hops": 9,
            "n_attn_heatmaps": 2,
            "initializer": tf.initializers.random_uniform(
                minval=-0.1, maxval=0.1
            ),
            "train_embeddings": False,
        }

    @classmethod
    def processing_fn(cls, features):
        return {
            "sentence": tf.sparse_concat(
                sp_inputs=[
                    features["left"],
                    features["target"],
                    features["right"],
                ],
                axis=1,
            ),
            "sentence_ids": tf.sparse_concat(
                sp_inputs=[
                    features["left_ids"],
                    features["target_ids"],
                    features["right_ids"],
                ],
                axis=1,
            ),
            "target": features["target"],
            "target_ids": features["target_ids"],
            "target_offset": features["left"].dense_shape[1] + 1,
        }

    @addon([attn_heatmaps])
    def model_fn(self, features, labels, mode, params):
        target_offset = tf.cast(features["target_offset"], tf.int32)
        sentence_ids = sparse_sequences_to_dense(features["sentence_ids"])
        target_ids = sparse_sequences_to_dense(features["target_ids"])

        with tf.variable_scope("embedding_layer", reuse=True):
            embeddings = tf.get_variable("embeddings")

        sentence_embedded = tf.nn.embedding_lookup(embeddings, sentence_ids)
        target_embedded = tf.nn.embedding_lookup(embeddings, target_ids)
        sentence_len = seq_lengths(sentence_ids)
        target_len = seq_lengths(target_ids)

        batch_size = tf.shape(sentence_embedded)[0]
        max_seq_len = tf.shape(sentence_embedded)[1]

        forward_cells = []
        backward_cells = []
        for _ in range(params["n_lstm_layers"]):
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
                inputs=sentence_embedded,
                sequence_length=sentence_len,
                dtype=tf.float32,
            )

        distances = get_bounded_distance_vectors(
            left_bounds=target_offset,
            right_bounds=target_offset + target_len,
            seq_lens=sentence_len,
            max_seq_len=max_seq_len,
        )

        u_t, w_t = calculate_u_t_w_t(sentence_len, max_seq_len, distances)

        memory = get_location_weighted_memory(memory_star, w_t, u_t)

        episode_0 = tf.zeros(shape=[batch_size, 1, params["gru_hidden_units"]])

        target_avg = variable_len_batch_mean(
            input_tensor=target_embedded,
            seq_lengths=target_len,
            op_name="target_avg_pooling",
        )

        attn_snapshots = create_snapshots_container(
            shape_like=sentence_ids, n_snaps=params["n_hops"]
        )

        attn_layer_num = tf.constant(1)

        initial_layer_inputs = (attn_layer_num, episode_0, attn_snapshots)

        def condition(attn_layer_num, episode, attn_snapshots):
            return tf.less_equal(attn_layer_num, params["n_hops"])

        def attn_layer_run(attn_layer_num, episode, attn_snapshots):
            mem_prev_ep_v_target = var_len_concatenate(
                seq_lens=sentence_len,
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
                    seq_lens=sentence_len,
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
                    sequence_length=sentence_len,
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
            cond=condition, body=attn_layer_run, loop_vars=initial_layer_inputs
        )

        literals, attn_snapshots = zip_attn_snapshots_with_sp_literals(
            sp_literals=features["sentence"],
            snapshots=attn_snapshots,
            num_layers=params["n_hops"],
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


def get_bounded_distance_vectors(
    left_bounds, right_bounds, seq_lens, max_seq_len
):
    batch_size = tf.shape(seq_lens)[0]
    mask = tf.sequence_mask(lengths=seq_lens, maxlen=max_seq_len)

    seq_range = tf.range(start=1, limit=max_seq_len + 1)
    seq_range_tiled = tf.tile([seq_range], multiples=[batch_size, 1])

    left_bounds = tf.expand_dims(left_bounds, axis=1)
    right_bounds = tf.expand_dims(right_bounds, axis=1)
    left_bounds_tiled = tf.ones_like(seq_range_tiled) * left_bounds
    right_bounds_tiled = tf.ones_like(seq_range_tiled) * right_bounds

    distance_left = tf.where(
        tf.less_equal(seq_range_tiled, left_bounds_tiled),
        left_bounds_tiled,
        seq_range_tiled,
    )
    distance_left_right = tf.where(
        tf.greater(distance_left, right_bounds_tiled),
        right_bounds_tiled,
        distance_left,
    )

    distances = seq_range_tiled - distance_left_right

    distances_masked = tf.where(mask, distances, tf.zeros_like(distances))

    return distances_masked


def calculate_u_t_w_t(seq_lens, max_seq_len, distances):
    mask = tf.sequence_mask(seq_lens, maxlen=max_seq_len, dtype=tf.float32)

    seq_len = tf.expand_dims(seq_lens, axis=1)

    u_t = tf.divide(distances, seq_len)

    u_t = tf.cast(u_t, tf.float32)

    w_t = (1 - tf.abs(u_t)) * mask

    return u_t, w_t


def get_location_weighted_memory(memory_star, w_t, u_t):
    u_t = tf.expand_dims(u_t, axis=2)
    w_t = tf.expand_dims(w_t, axis=2)

    memory = tf.multiply(memory_star, w_t)
    memory = tf.concat([memory, u_t], axis=2)

    return memory


def var_len_concatenate(seq_lens, memory, v_target, prev_episode):
    max_seq_len = tf.shape(memory)[1]

    mask = tf.sequence_mask(seq_lens, maxlen=max_seq_len, dtype=tf.float32)
    mask = tf.expand_dims(mask, axis=2)

    prev_episode = tf.tile(prev_episode, multiples=[1, max_seq_len, 1])
    v_target = tf.tile(v_target, multiples=[1, max_seq_len, 1])

    mem_prev_e = tf.concat([memory, prev_episode * mask], axis=2)
    mem_prev_e_v_t = tf.concat([mem_prev_e, v_target * mask], axis=2)

    return mem_prev_e_v_t


def ram_attn_unit(seq_lens, attn_focus, weight_dim, init):
    batch_size = tf.shape(attn_focus)[0]
    max_seq_len = tf.shape(attn_focus)[1]
    w_att = tf.get_variable(
        name="weights",
        shape=[1, weight_dim],
        dtype=tf.float32,
        initializer=init,
    )
    b_att = tf.get_variable(
        name="bias", shape=[1], dtype=tf.float32, initializer=init
    )

    w_att_batch_dim = tf.expand_dims(w_att, axis=0)
    w_att_tiled = tf.tile(
        w_att_batch_dim, multiples=[batch_size * max_seq_len, 1, 1]
    )
    w_att_batched = tf.reshape(
        w_att_tiled, shape=[-1, max_seq_len, 1, weight_dim]
    )

    b_att_mask = tf.sequence_mask(
        seq_lens, maxlen=max_seq_len, dtype=tf.float32
    )
    b_att_masked = b_att_mask * b_att
    b_att_batched = tf.reshape(b_att_masked, shape=[batch_size, -1, 1, 1])

    attn_focus_transpose = tf.expand_dims(attn_focus, axis=3)
    g_score = tf.einsum("Baij,Bajk->Baik", w_att_batched, attn_focus_transpose)
    g_score = g_score + b_att_batched

    g_score = tf.squeeze(g_score, axis=3)

    softmax_mask = tf.cast(b_att_mask, dtype=tf.bool)

    attn_weights = masked_softmax(logits=g_score, mask=softmax_mask)

    return attn_weights
