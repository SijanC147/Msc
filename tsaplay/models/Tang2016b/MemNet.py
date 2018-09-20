import tensorflow as tf
from tensorflow.estimator import (  # pylint: disable=E0401
    EstimatorSpec,
    ModeKeys,
)
from tsaplay.models.Model import Model
from tsaplay.models.Tang2016b.common import (
    params as default_params,
    memnet_input_fn,
    get_absolute_distance_vector,
    get_location_vector_model,
    content_attention_model,
    zip_hop_attn_snapshots_with_literals,
)
from tsaplay.utils._tf import (
    variable_len_batch_mean,
    generate_attn_heatmap_summary,
)


class MemNet(Model):
    def _params(self):
        return default_params

    def _feature_columns(self):
        default = []
        return default

    def _train_input_fn(self):
        return lambda features, labels, batch_size: memnet_input_fn(
            features,
            labels,
            batch_size,
            max_seq_length=self.params["max_seq_length"],
        )

    def _eval_input_fn(self):
        return lambda features, labels, batch_size: memnet_input_fn(
            features,
            labels,
            batch_size,
            max_seq_length=self.params["max_seq_length"],
            eval_input=True,
        )

    def _model_fn(self):
        def default(features, labels, mode, params=self.params):
            with tf.variable_scope("embedding_layer", reuse=tf.AUTO_REUSE):
                embeddings = tf.get_variable(
                    "embeddings",
                    shape=[params["vocab_size"], params["embedding_dim"]],
                    initializer=params["embedding_initializer"],
                    trainable=False,
                )

            m = tf.contrib.layers.embed_sequence(
                ids=features["context"]["x"],
                initializer=embeddings,
                scope="embedding_layer",
                reuse=True,
            )

            context_locations = get_absolute_distance_vector(
                target_locs=features["target"]["loc"],
                seq_lens=features["context"]["len"],
                max_seq_len=params["max_seq_length"],
            )

            target_embeddings = tf.contrib.layers.embed_sequence(
                ids=features["target"]["x"],
                initializer=embeddings,
                scope="embedding_layer",
                reuse=True,
            )

            v_aspect = variable_len_batch_mean(
                input_tensor=target_embeddings,
                seq_lengths=features["target"]["len"],
                op_name="target_embedding_avg",
            )

            def condition(hop_num, input_vec, ext_memory, attn_snapshots):
                return tf.less_equal(hop_num, params["n_hops"])

            def hop(hop_num, input_vec, ext_memory, attn_snapshots):
                location_vector_model_fn = get_location_vector_model(
                    model_num=params["location_model"]
                )

                v_loc = location_vector_model_fn(
                    locs=context_locations,
                    seq_lens=features["context"]["len"],
                    emb_dim=params["embedding_dim"],
                    hop=hop_num,
                    init=params["initializer"],
                )

                if params["location_model"] == 3:
                    ext_memory = ext_memory + v_loc
                else:
                    ext_memory = tf.multiply(m, v_loc)

                with tf.variable_scope("linear_layer", reuse=tf.AUTO_REUSE):
                    linear_out = tf.layers.dense(
                        inputs=tf.squeeze(input_vec, axis=1),
                        units=params["embedding_dim"],
                        activation=None,
                        kernel_initializer=params["initializer"],
                        bias_initializer=params["initializer"],
                    )

                with tf.variable_scope("attention_layer", reuse=tf.AUTO_REUSE):
                    attn_out, attn_snapshot = content_attention_model(
                        seq_lens=features["context"]["len"],
                        memory=ext_memory,
                        v_aspect=input_vec,
                        emb_dim=params["embedding_dim"],
                        init=params["initializer"],
                        literal=features["context"]["lit"],
                    )

                attn_snapshot.set_shape([None, params["max_seq_length"], 1])
                attn_snapshot = tf.expand_dims(attn_snapshot, axis=0)
                batch_diff = params["batch_size"] - tf.shape(attn_snapshot)[1]
                attn_snapshot = tf.pad(
                    attn_snapshot,
                    paddings=[
                        [hop_num - 1, params["n_hops"] - hop_num],
                        [0, batch_diff],
                        [0, 0],
                        [0, 0],
                    ],
                )
                attn_snapshots = tf.add(attn_snapshots, attn_snapshot)

                output_vec = attn_out + linear_out
                output_vec = tf.expand_dims(output_vec, axis=1)

                hop_num = tf.add(hop_num, 1)

                return (hop_num, output_vec, ext_memory, attn_snapshots)

            attn_snapshots = tf.get_variable(
                name="attn",
                shape=[
                    params["n_hops"],
                    params["batch_size"],
                    params["max_seq_length"],
                    1,
                ],
                dtype=tf.float32,
                initializer=tf.initializers.zeros,
                trainable=False,
            )

            hop_number = tf.constant(1)

            initial_hop_inputs = (hop_number, v_aspect, m, attn_snapshots)

            _, final_sentence_rep, _, attn_snapshots = tf.while_loop(
                cond=condition,
                body=hop,
                loop_vars=initial_hop_inputs,
                shape_invariants=(
                    hop_number.get_shape(),
                    v_aspect.get_shape(),
                    m.get_shape(),
                    tf.TensorShape(dims=[params["n_hops"], None, None, 1]),
                ),
            )

            literals, attn_snapshots = zip_hop_attn_snapshots_with_literals(
                literals=features["context"]["lit"],
                snapshots=attn_snapshots,
                max_len=params["max_seq_length"],
                num_hops=params["n_hops"],
            )
            attn_info = tf.tuple([literals, attn_snapshots])
            generate_attn_heatmap_summary(attn_info)

            final_sentence_rep = tf.squeeze(final_sentence_rep, axis=1)

            logits = tf.layers.dense(
                inputs=final_sentence_rep,
                units=params["n_out_classes"],
                kernel_initializer=params["initializer"],
                bias_initializer=params["initializer"],
            )

            predictions = {
                "class_ids": tf.argmax(logits, 1),
                "probabilities": tf.nn.softmax(logits),
                "logits": logits,
            }

            if mode == ModeKeys.PREDICT:
                return EstimatorSpec(mode, predictions=predictions)

            loss = tf.losses.sparse_softmax_cross_entropy(
                labels=labels, logits=logits
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
