import tensorflow as tf
from tensorflow.contrib.rnn import (  # pylint: disable=E0611
    stack_bidirectional_dynamic_rnn
)
from tsaplay.models.Model import Model
from tsaplay.models.Zheng2018.common import (
    params as default_params,
    lcr_rot_input_fn,
    dropout_lstm_cell,
)
from tsaplay.utils import variable_len_batch_mean


class LcrRot(Model):
    def _params(self):
        return default_params

    def _feature_columns(self):
        default = []
        return default

    def _train_input_fn(self):
        return lambda features, labels, batch_size: lcr_rot_input_fn(
            features,
            labels,
            batch_size,
            max_seq_length=self.params["max_seq_length"],
        )

    def _eval_input_fn(self):
        return lambda features, labels, batch_size: lcr_rot_input_fn(
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
                )

            left_embeddings = tf.contrib.layers.embed_sequence(
                ids=features["left"]["x"],
                initializer=embeddings,
                scope="embedding_layer",
                reuse=True,
            )

            target_embeddings = tf.contrib.layers.embed_sequence(
                ids=features["target"]["x"],
                initializer=embeddings,
                scope="embedding_layer",
                reuse=True,
            )

            right_embeddings = tf.contrib.layers.embed_sequence(
                ids=features["right"]["x"],
                initializer=embeddings,
                scope="embedding_layer",
                reuse=True,
            )

            with tf.variable_scope("target_bi_lstm", reuse=tf.AUTO_REUSE):
                target_hidden_states, _, _ = stack_bidirectional_dynamic_rnn(
                    cells_fw=[dropout_lstm_cell(params)],
                    cells_bw=[dropout_lstm_cell(params)],
                    inputs=target_embeddings,
                    sequence_length=features["target"]["len"],
                    dtype=tf.float32,
                )
                target_pooled = variable_len_batch_mean(
                    input_tensor=target_hidden_states,
                    seq_lengths=features["target"]["len"],
                    op_name="target_avg_pooling",
                )

            with tf.variable_scope("left_bi_lstm", reuse=tf.AUTO_REUSE):
                left_hidden_states, _, _ = stack_bidirectional_dynamic_rnn(
                    cells_fw=[dropout_lstm_cell(params)],
                    cells_bw=[dropout_lstm_cell(params)],
                    inputs=left_embeddings,
                    sequence_length=features["left"]["len"],
                    dtype=tf.float32,
                )
            print(left_hidden_states)

            with tf.variable_scope("right_bi_lstm", reuse=tf.AUTO_REUSE):
                right_hidden_states, _, _ = stack_bidirectional_dynamic_rnn(
                    cells_fw=[dropout_lstm_cell(params)],
                    cells_bw=[dropout_lstm_cell(params)],
                    inputs=right_embeddings,
                    sequence_length=features["right"]["len"],
                    dtype=tf.float32,
                )
            print(right_hidden_states)

            with tf.name_scope("target_to_context_Attn"):
                with tf.variable_scope("left"):
                    Wlc = tf.get_variable(
                        name="W_l_c",
                        shape=[params["hidden_units"] * 2, 1],
                        dtype=tf.float32,
                        initializer=tf.random_uniform_initializer(
                            minval=-0.1, maxval=0.1
                        ),
                    )
                    blc = tf.get_variable(
                        name="b_l_c",
                        shape=[1],
                        dtype=tf.float32,
                        initializer=tf.random_uniform_initializer(
                            minval=-0.1, maxval=0.1
                        ),
                    )
                    score_fn = tf.tanh(
                        tf.matmul(
                            tf.matmul(left_hidden_states, Wlc),
                            target_pooled,
                            transpose_b=True,
                        )
                        + blc
                    )

            # concatenated_final_states = tf.concat(
            #     [
            #         target_pooled
            #         # left_hidden_states,
            #         # target_hidden_states,
            #         # right_hidden_states,
            #     ],
            #     axis=1,
            # )

            # logits = tf.layers.dense(
            #     inputs=concatenated_final_states, units=params["n_out_classes"]
            # )

            concat_final_states = tf.concat([target_pooled], axis=1)

            logits = tf.layers.dense(inputs=concat_final_states, units=1)

            print(logits)
            predicted_classes = tf.argmax(logits, 1)
            print(predicted_classes)

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
                labels=labels, predictions=predicted_classes, name="acc_op"
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
                {"loss": loss, "accuracy": accuracy[1]}, every_n_iter=100
            )

            return tf.estimator.EstimatorSpec(
                mode,
                loss=loss,
                train_op=train_op,
                training_hooks=[logging_hook],
            )

        return default
