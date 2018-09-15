import tensorflow as tf
from tsaplay.models.Model import Model
from tsaplay.models.Ma2017.common import (
    params as default_params,
    ian_input_fn,
    dropout_lstm_cell,
    attention_unit,
)
from tsaplay.utils.common import variable_len_batch_mean


class InteractiveAttentionNetwork(Model):
    def _params(self):
        return default_params

    def _feature_columns(self):
        default = []
        return default

    def _train_input_fn(self):
        return lambda features, labels, batch_size: ian_input_fn(
            features,
            labels,
            batch_size,
            max_seq_length=self.params["max_seq_length"],
        )

    def _eval_input_fn(self):
        return lambda features, labels, batch_size: ian_input_fn(
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

            context_embeddings = tf.contrib.layers.embed_sequence(
                ids=features["context"]["x"],
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

            with tf.variable_scope("context_lstm"):
                context_hidden_states, _ = tf.nn.dynamic_rnn(
                    cell=dropout_lstm_cell(params),
                    inputs=context_embeddings,
                    sequence_length=features["context"]["len"],
                    dtype=tf.float32,
                )
                c_avg = variable_len_batch_mean(
                    input_tensor=context_hidden_states,
                    seq_lengths=features["context"]["len"],
                    op_name="context_avg_pooling",
                )

            with tf.variable_scope("target_lstm"):
                target_hidden_states, _ = tf.nn.dynamic_rnn(
                    cell=dropout_lstm_cell(params),
                    inputs=target_embeddings,
                    sequence_length=features["target"]["len"],
                    dtype=tf.float32,
                )
                t_avg = variable_len_batch_mean(
                    input_tensor=target_hidden_states,
                    seq_lengths=features["target"]["len"],
                    op_name="target_avg_pooling",
                )

            with tf.variable_scope("attention_layer", reuse=tf.AUTO_REUSE):
                c_r = attention_unit(
                    h_states=context_hidden_states,
                    hidden_units=params["hidden_units"],
                    seq_lengths=features["context"]["len"],
                    attn_focus=t_avg,
                    init=params["initializer"],
                )
                t_r = attention_unit(
                    h_states=target_hidden_states,
                    hidden_units=params["hidden_units"],
                    seq_lengths=features["target"]["len"],
                    attn_focus=c_avg,
                    init=params["initializer"],
                )

                final_sentence_rep = tf.concat([t_r, c_r], axis=1)

                logits = tf.layers.dense(
                    inputs=final_sentence_rep,
                    units=params["n_out_classes"],
                    activation=tf.nn.tanh,
                    kernel_initializer=params["initializer"],
                    bias_initializer=params["initializer"],
                )

            predicted_classes = tf.argmax(logits, 1)

            predictions = {
                "class_ids": predicted_classes,
                "probabilities": tf.nn.softmax(logits),
                "logits": logits,
            }

            if mode == tf.estimator.ModeKeys.PREDICT:
                return tf.estimator.EstimatorSpec(
                    mode, predictions=predictions
                )

            loss = tf.losses.sparse_softmax_cross_entropy(
                labels=labels, logits=logits
            )
            l2_reg = tf.reduce_sum(
                [tf.nn.l2_loss(v) for v in tf.trainable_variables()]
            )
            loss = loss + params["l2_weight"] * l2_reg

            accuracy = tf.metrics.accuracy(
                labels=labels, predictions=predicted_classes, name="acc_op"
            )
            tf.summary.scalar("accuracy", accuracy[1])
            tf.summary.scalar("loss", loss)

            if mode == tf.estimator.ModeKeys.EVAL:
                return tf.estimator.EstimatorSpec(
                    mode,
                    loss=loss,
                    predictions=predictions,
                    eval_metric_ops={"accuracy": accuracy},
                )

            optimizer = tf.train.MomentumOptimizer(
                learning_rate=params["learning_rate"],
                momentum=params["momentum"],
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
                predictions=predictions,
                training_hooks=[logging_hook],
            )

        return default
