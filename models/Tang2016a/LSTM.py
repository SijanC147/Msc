import tensorflow as tf
from models.Model import Model
from models.Tang2016a import common


class LSTM(Model):
    def _params(self):
        return common.params

    def _feature_columns(self):
        return []

    def _train_input_fn(self):
        return lambda features, labels, batch_size: common.lstm_input_fn(
            features,
            labels,
            batch_size,
            max_seq_length=self.params["max_seq_length"],
        )

    def _eval_input_fn(self):
        return lambda features, labels, batch_size: common.lstm_input_fn(
            features,
            labels,
            batch_size,
            max_seq_length=self.params["max_seq_length"],
            eval_input=True,
        )

    def _model_fn(self):
        def _default(features, labels, mode, params=self.params):
            inputs = tf.contrib.layers.embed_sequence(
                ids=features["x"],
                vocab_size=params["vocab_size"],
                embed_dim=params["embedding_dim"],
                initializer=params["embedding_initializer"],
            )

            _, final_states = tf.nn.dynamic_rnn(
                cell=common.dropout_lstm_cell(params),
                inputs=inputs,
                sequence_length=features["len"],
                dtype=tf.float32,
            )

            logits = tf.layers.dense(
                inputs=final_states.h, units=params["n_out_classes"]
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

    # def set_params(self, params):
    #     default = common.params
    #     super().set_params(default if params is None else params)

    # def set_feature_columns(self, feature_columns):
    #     default = []
    #     super().set_feature_columns(
    #         default if feature_columns is None else feature_columns
    #     )

    # def set_train_input_fn(self, train_input_fn):
    #     def _default(
    #         features, labels, batch_size=self.params.get("batch_size")
    #     ):
    #         return common.lstm_input_fn(
    #             features,
    #             labels,
    #             batch_size,
    #             max_seq_length=self.params["max_seq_length"],
    #         )

    #     super().set_train_input_fn(
    #         _default if train_input_fn is None else train_input_fn
    #     )

    # def set_eval_input_fn(self, eval_input_fn):
    #     def _default(
    #         features, labels, batch_size=self.params.get("batch_size")
    #     ):
    #         return common.lstm_input_fn(
    #             features,
    #             labels,
    #             batch_size,
    #             max_seq_length=self.params["max_seq_length"],
    #             eval_input=True,
    #         )

    #     super().set_eval_input_fn(
    #         _default if eval_input_fn is None else eval_input_fn
    #     )
