import tensorflow as tf
from tensorflow.estimator import (  # pylint: disable=E0401
    EstimatorSpec,
    ModeKeys,
)
from tensorflow.contrib.rnn import (  # pylint: disable=E0611
    stack_bidirectional_dynamic_rnn
)
from tsaplay.models.Model import Model
from tsaplay.models.Zheng2018.common import (
    params as default_params,
    lcr_rot_input_fn,
)
from tsaplay.utils._tf import (
    variable_len_batch_mean,
    attention_unit,
    dropout_lstm_cell,
    l2_regularized_loss,
)


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
                    trainable=False,
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

            with tf.variable_scope("target_bi_lstm"):
                target_hidden_states, _, _ = stack_bidirectional_dynamic_rnn(
                    cells_fw=[
                        dropout_lstm_cell(
                            hidden_units=params["hidden_units"],
                            initializer=params["initializer"],
                            keep_prob=params["keep_prob"],
                        )
                    ],
                    cells_bw=[
                        dropout_lstm_cell(
                            hidden_units=params["hidden_units"],
                            initializer=params["initializer"],
                            keep_prob=params["keep_prob"],
                        )
                    ],
                    inputs=target_embeddings,
                    sequence_length=features["target"]["len"],
                    dtype=tf.float32,
                )
                r_t = variable_len_batch_mean(
                    input_tensor=target_hidden_states,
                    seq_lengths=features["target"]["len"],
                    op_name="target_avg_pooling",
                )

            with tf.variable_scope("left_bi_lstm"):
                left_hidden_states, _, _ = stack_bidirectional_dynamic_rnn(
                    cells_fw=[
                        dropout_lstm_cell(
                            hidden_units=params["hidden_units"],
                            initializer=params["initializer"],
                            keep_prob=params["keep_prob"],
                        )
                    ],
                    cells_bw=[
                        dropout_lstm_cell(
                            hidden_units=params["hidden_units"],
                            initializer=params["initializer"],
                            keep_prob=params["keep_prob"],
                        )
                    ],
                    inputs=left_embeddings,
                    sequence_length=features["left"]["len"],
                    dtype=tf.float32,
                )

            with tf.variable_scope("right_bi_lstm"):
                right_hidden_states, _, _ = stack_bidirectional_dynamic_rnn(
                    cells_fw=[
                        dropout_lstm_cell(
                            hidden_units=params["hidden_units"],
                            initializer=params["initializer"],
                            keep_prob=params["keep_prob"],
                        )
                    ],
                    cells_bw=[
                        dropout_lstm_cell(
                            hidden_units=params["hidden_units"],
                            initializer=params["initializer"],
                            keep_prob=params["keep_prob"],
                        )
                    ],
                    inputs=right_embeddings,
                    sequence_length=features["right"]["len"],
                    dtype=tf.float32,
                )

            with tf.variable_scope("left_t2c_attn"):
                r_l = attention_unit(
                    h_states=left_hidden_states,
                    hidden_units=params["hidden_units"] * 2,
                    seq_lengths=features["left"]["len"],
                    attn_focus=r_t,
                    init=params["initializer"],
                )

            with tf.variable_scope("right_t2c_attn"):
                r_r = attention_unit(
                    h_states=right_hidden_states,
                    hidden_units=params["hidden_units"] * 2,
                    seq_lengths=features["right"]["len"],
                    attn_focus=r_t,
                    init=params["initializer"],
                )

            with tf.variable_scope("left_c2t_attn"):
                r_t_l = attention_unit(
                    h_states=target_hidden_states,
                    hidden_units=params["hidden_units"] * 2,
                    seq_lengths=features["target"]["len"],
                    attn_focus=tf.expand_dims(r_l, axis=1),
                    init=params["initializer"],
                )

            with tf.variable_scope("right_c2t_attn"):
                r_t_r = attention_unit(
                    h_states=target_hidden_states,
                    hidden_units=params["hidden_units"] * 2,
                    seq_lengths=features["target"]["len"],
                    attn_focus=tf.expand_dims(r_r, axis=1),
                    init=params["initializer"],
                )

            final_sentence_rep = tf.concat([r_l, r_t_l, r_t_r, r_r], axis=1)

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

            optimizer = tf.train.MomentumOptimizer(
                learning_rate=params["learning_rate"],
                momentum=params["momentum"],
            )
            train_op = optimizer.minimize(
                loss, global_step=tf.train.get_global_step()
            )

            return EstimatorSpec(
                mode, loss=loss, train_op=train_op, predictions=predictions
            )

        return default
