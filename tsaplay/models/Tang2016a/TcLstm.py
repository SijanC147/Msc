import tensorflow as tf
from tensorflow.estimator import (  # pylint: disable=E0401
    EstimatorSpec,
    ModeKeys,
)
from tsaplay.models.Model import Model
from tsaplay.models.Tang2016a.common import (
    params as default_params,
    tclstm_input_fn,
)
from tsaplay.utils._tf import variable_len_batch_mean, dropout_lstm_cell


class TcLstm(Model):
    def _params(self):
        return default_params

    def _feature_columns(self):
        return []

    def _train_input_fn(self):
        return lambda features, labels, batch_size: tclstm_input_fn(
            features,
            labels,
            batch_size,
            max_seq_length=self.params["max_seq_length"],
        )

    def _eval_input_fn(self):
        return lambda features, labels, batch_size: tclstm_input_fn(
            features,
            labels,
            batch_size,
            max_seq_length=self.params["max_seq_length"],
            eval_input=True,
        )

    def _model_fn(self):
        def _default(features, labels, mode, params=self.params):
            with tf.variable_scope("embedding_layer", reuse=tf.AUTO_REUSE):
                embeddings = tf.get_variable(
                    "embeddings",
                    shape=[params["vocab_size"], params["embedding_dim"]],
                    initializer=params["embedding_initializer"],
                )

            target_embedding = tf.contrib.layers.embed_sequence(
                ids=features["target"]["x"],
                initializer=embeddings,
                scope="embedding_layer",
                reuse=True,
            )

            left_inputs = tf.contrib.layers.embed_sequence(
                ids=features["left"]["x"],
                initializer=embeddings,
                scope="embedding_layer",
                reuse=True,
            )

            right_inputs = tf.contrib.layers.embed_sequence(
                ids=features["right"]["x"],
                initializer=embeddings,
                scope="embedding_layer",
                reuse=True,
            )

            with tf.name_scope("target_connection"):
                mean_target_embedding = variable_len_batch_mean(
                    input_tensor=target_embedding,
                    seq_lengths=features["target"]["len"],
                    op_name="target_embedding_avg",
                )
                left_inputs = tf.stack(
                    values=[
                        left_inputs,
                        tf.ones(tf.shape(left_inputs)) * mean_target_embedding,
                    ],
                    axis=2,
                )
                left_inputs = tf.reshape(
                    tensor=left_inputs,
                    shape=[
                        -1,
                        params["max_seq_length"],
                        2 * params["embedding_dim"],
                    ],
                )
                right_inputs = tf.stack(
                    values=[
                        right_inputs,
                        tf.ones(tf.shape(right_inputs))
                        * mean_target_embedding,
                    ],
                    axis=2,
                )
                right_inputs = tf.reshape(
                    tensor=right_inputs,
                    shape=[
                        -1,
                        params["max_seq_length"],
                        2 * params["embedding_dim"],
                    ],
                )

            with tf.variable_scope("left_lstm"):
                _, final_states_left = tf.nn.dynamic_rnn(
                    cell=dropout_lstm_cell(
                        hidden_units=params["hidden_units"],
                        initializer=params["initializer"],
                        keep_prob=params["keep_prob"],
                    ),
                    inputs=left_inputs,
                    sequence_length=features["left"]["len"],
                    dtype=tf.float32,
                )

            with tf.variable_scope("right_lstm"):
                _, final_states_right = tf.nn.dynamic_rnn(
                    cell=dropout_lstm_cell(
                        hidden_units=params["hidden_units"],
                        initializer=params["initializer"],
                        keep_prob=params["keep_prob"],
                    ),
                    inputs=right_inputs,
                    sequence_length=features["right"]["len"],
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

        return _default
