import tensorflow as tf
from tensorflow.estimator import (  # pylint: disable=E0401
    EstimatorSpec,
    ModeKeys,
)
from tsaplay.models.TSAModel import TSAModel
from tsaplay.utils.tf import (
    sparse_sequences_to_dense,
    sparse_reverse,
    seq_lengths,
    variable_len_batch_mean,
    dropout_lstm_cell,
    setup_embedding_layer,
    get_embedded_seq,
)


class TCLstm(TSAModel):
    def set_params(self):
        return {
            "batch_size": 100,
            "n_out_classes": 3,
            "learning_rate": 0.01,
            "keep_prob": 0.8,
            "hidden_units": 50,
            "initializer": tf.initializers.random_uniform(
                minval=-0.03, maxval=0.03
            ),
        }

    @classmethod
    def processing_fn(cls, features):
        return {
            "left_ids": tf.sparse_concat(
                sp_inputs=[features["left_ids"], features["target_ids"]],
                axis=1,
            ),
            "right_ids": sparse_reverse(
                tf.sparse_concat(
                    sp_inputs=[features["right_ids"], features["target_ids"]],
                    axis=1,
                )
            ),
            "target_ids": features["target_ids"],
        }

    def model_fn(self, features, labels, mode, params):
        left_ids = sparse_sequences_to_dense(features["left_ids"])
        target_ids = sparse_sequences_to_dense(features["target_ids"])
        right_ids = sparse_sequences_to_dense(features["right_ids"])
        left_len = seq_lengths(left_ids)
        target_len = seq_lengths(target_ids)
        right_len = seq_lengths(right_ids)

        embedding_matrix = setup_embedding_layer(
            vocab_size=params["vocab_size"],
            dim_size=params["embedding_dim"],
            init=params["embedding_initializer"],
        )

        left_embeddings = get_embedded_seq(left_ids, embedding_matrix)
        target_embeddings = get_embedded_seq(target_ids, embedding_matrix)
        right_embeddings = get_embedded_seq(right_ids, embedding_matrix)

        max_left_len = tf.shape(left_embeddings)[1]
        max_right_len = tf.shape(right_embeddings)[1]

        with tf.name_scope("target_connection"):
            mean_target_embedding = variable_len_batch_mean(
                input_tensor=target_embeddings,
                seq_lengths=target_len,
                op_name="target_embedding_avg",
            )
            left_embeddings = tf.stack(
                values=[
                    left_embeddings,
                    tf.ones(tf.shape(left_embeddings)) * mean_target_embedding,
                ],
                axis=2,
            )
            left_embeddings = tf.reshape(
                tensor=left_embeddings,
                shape=[-1, max_left_len, 2 * params["embedding_dim"]],
            )
            right_embeddings = tf.stack(
                values=[
                    right_embeddings,
                    tf.ones(tf.shape(right_embeddings))
                    * mean_target_embedding,
                ],
                axis=2,
            )
            right_embeddings = tf.reshape(
                tensor=right_embeddings,
                shape=[-1, max_right_len, 2 * params["embedding_dim"]],
            )

        with tf.variable_scope("left_lstm"):
            _, final_states_left = tf.nn.dynamic_rnn(
                cell=dropout_lstm_cell(
                    hidden_units=params["hidden_units"],
                    initializer=params["initializer"],
                    keep_prob=params["keep_prob"],
                ),
                inputs=left_embeddings,
                sequence_length=left_len,
                dtype=tf.float32,
            )

        with tf.variable_scope("right_lstm"):
            _, final_states_right = tf.nn.dynamic_rnn(
                cell=dropout_lstm_cell(
                    hidden_units=params["hidden_units"],
                    initializer=params["initializer"],
                    keep_prob=params["keep_prob"],
                ),
                inputs=right_embeddings,
                sequence_length=right_len,
                dtype=tf.float32,
            )

        concatenated_final_states = tf.concat(
            [final_states_left.h, final_states_right.h], axis=1
        )

        logits = tf.layers.dense(
            inputs=concatenated_final_states, units=params["n_out_classes"]
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

        optimizer = tf.train.AdagradOptimizer(
            learning_rate=params["learning_rate"]
        )
        train_op = optimizer.minimize(
            loss, global_step=tf.train.get_global_step()
        )

        return EstimatorSpec(
            mode, loss=loss, train_op=train_op, predictions=predictions
        )
