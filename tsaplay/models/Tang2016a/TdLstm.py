import tensorflow as tf
from tensorflow.estimator import (  # pylint: disable=E0401
    EstimatorSpec,
    ModeKeys,
)
from tsaplay.models.Model import Model
from tsaplay.models.Tang2016a.common import (
    params as default_params,
    tdlstm_input_fn,
    tdlstm_serving_fn,
)
from tsaplay.utils._tf import (
    sparse_seq_lengths,
    dropout_lstm_cell,
    setup_embedding_layer,
    get_embedded_seq,
)


class TdLstm(Model):
    def _params(self):
        return default_params

    def _feature_columns(self):
        return []

    def _train_input_fn(self):
        return lambda tfrecord, batch_size: tdlstm_input_fn(
            tfrecord, batch_size
        )

    def _eval_input_fn(self):
        return lambda tfrecord, batch_size: tdlstm_input_fn(
            tfrecord, batch_size, _eval=True
        )

    def _serving_input_fn(self):
        return lambda features: tdlstm_serving_fn(features)

    def _model_fn(self):
        def _default(features, labels, mode, params=self.params):
            left_len = sparse_seq_lengths(features["left_ids"])
            right_len = sparse_seq_lengths(features["right_ids"])
            left_ids = tf.sparse_tensor_to_dense(features["left_ids"])
            right_ids = tf.sparse_tensor_to_dense(features["right_ids"])
            if mode == ModeKeys.TRAIN or mode == ModeKeys.EVAL:
                left_ids = tf.squeeze(left_ids, axis=1)
                right_ids = tf.squeeze(right_ids, axis=1)

            embedding_matrix = setup_embedding_layer(
                vocab_size=params["vocab_size"],
                dim_size=params["embedding_dim"],
                init=params["embedding_initializer"],
            )

            left_inputs = get_embedded_seq(left_ids, embedding_matrix)
            right_inputs = get_embedded_seq(right_ids, embedding_matrix)

            with tf.variable_scope("left_lstm"):
                _, final_states_left = tf.nn.dynamic_rnn(
                    cell=dropout_lstm_cell(
                        hidden_units=params["hidden_units"],
                        initializer=params["initializer"],
                        keep_prob=params["keep_prob"],
                    ),
                    inputs=left_inputs,
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
                    inputs=right_inputs,
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

        return _default
