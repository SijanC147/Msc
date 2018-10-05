import tensorflow as tf
from tensorflow.estimator import (  # pylint: disable=E0401
    EstimatorSpec,
    ModeKeys,
)
from tsaplay.models.TSAModel import TSAModel
from tsaplay.utils.tf import (
    sparse_reverse,
    variable_len_batch_mean,
    dropout_lstm_cell,
)
from tsaplay.utils.decorators import prep_features


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

    @prep_features(["left", "target", "right"])
    def model_fn(self, features, labels, mode, params):
        max_left_len = tf.shape(features["left_emb"])[1]
        max_right_len = tf.shape(features["right_emb"])[1]

        with tf.name_scope("target_connection"):
            mean_target_embedding = variable_len_batch_mean(
                input_tensor=features["target_emb"],
                seq_lengths=features["target_len"],
                op_name="target_embedding_avg",
            )
            features["left_emb"] = tf.stack(
                values=[
                    features["left_emb"],
                    tf.ones(tf.shape(features["left_emb"]))
                    * mean_target_embedding,
                ],
                axis=2,
            )
            features["left_emb"] = tf.reshape(
                tensor=features["left_emb"],
                shape=[-1, max_left_len, 2 * params["embedding_dim"]],
            )
            features["right_emb"] = tf.stack(
                values=[
                    features["right_emb"],
                    tf.ones(tf.shape(features["right_emb"]))
                    * mean_target_embedding,
                ],
                axis=2,
            )
            features["right_emb"] = tf.reshape(
                tensor=features["right_emb"],
                shape=[-1, max_right_len, 2 * params["embedding_dim"]],
            )

        with tf.variable_scope("left_lstm"):
            _, final_states_left = tf.nn.dynamic_rnn(
                cell=dropout_lstm_cell(
                    hidden_units=params["hidden_units"],
                    initializer=params["initializer"],
                    keep_prob=params["keep_prob"],
                ),
                inputs=features["left_emb"],
                sequence_length=features["left_len"],
                dtype=tf.float32,
            )

        with tf.variable_scope("right_lstm"):
            _, final_states_right = tf.nn.dynamic_rnn(
                cell=dropout_lstm_cell(
                    hidden_units=params["hidden_units"],
                    initializer=params["initializer"],
                    keep_prob=params["keep_prob"],
                ),
                inputs=features["right_emb"],
                sequence_length=features["right_len"],
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
