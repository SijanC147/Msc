import tensorflow as tf
from models.Model import Model
from models.Tang2016a.common import (
    params as default_params,
    dropout_lstm_cell,
    tdlstm_input_fn,
)


class TdLstm(Model):
    def _params(self):
        return default_params

    def _feature_columns(self):
        return []

    def _train_input_fn(self):
        return lambda features, labels, batch_size: tdlstm_input_fn(
            features,
            labels,
            batch_size,
            max_seq_length=self.params["max_seq_length"],
        )

    def _eval_input_fn(self):
        return lambda features, labels, batch_size: tdlstm_input_fn(
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

            with tf.variable_scope("left_lstm"):
                _, final_states_left = tf.nn.dynamic_rnn(
                    cell=dropout_lstm_cell(params),
                    inputs=left_inputs,
                    sequence_length=features["left"]["len"],
                    dtype=tf.float32,
                )

            with tf.variable_scope("right_lstm"):
                _, final_states_right = tf.nn.dynamic_rnn(
                    cell=dropout_lstm_cell(params),
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

            predicted_classes = tf.argmax(logits, 1)

            if mode == tf.estimator.ModeKeys.PREDICT:
                predictions = {
                    "class_ids": predicted_classes[:, tf.newaxis],
                    "probabilities": tf.nn.softmax(logits),
                    "logits": logits,
                }
                return tf.estimator.EstimatorSpec(
                    mode, predictions=predictions
                )

            loss = tf.losses.sparse_softmax_cross_entropy(
                labels=labels, logits=logits
            )
            accuracy = tf.metrics.accuracy(
                labels=labels, predictions=predicted_classes
            )

            metrics = {"accuracy": accuracy}

            tf.summary.scalar("accuracy", accuracy[1])
            tf.summary.scalar("loss", loss)

            if mode == tf.estimator.ModeKeys.EVAL:
                return tf.estimator.EstimatorSpec(
                    mode, loss=loss, eval_metric_ops=metrics
                )

            optimizer = tf.train.AdagradOptimizer(
                learning_rate=params["learning_rate"]
            )
            train_op = optimizer.minimize(
                loss, global_step=tf.train.get_global_step()
            )

            logging_hook = tf.train.LoggingTensorHook(
                {"loss": loss, "accuracy": accuracy[1]}, every_n_iter=50
            )

            return tf.estimator.EstimatorSpec(
                mode,
                loss=loss,
                train_op=train_op,
                training_hooks=[logging_hook],
            )

        return _default
