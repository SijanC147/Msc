import tensorflow as tf
from tensorflow.estimator import (  # pylint: disable=E0401
    EstimatorSpec,
    ModeKeys,
)
from tsaplay.models.Model import Model
from tsaplay.models.Ma2017.common import params as default_params, ian_input_fn
from tsaplay.utils._tf import (
    variable_len_batch_mean,
    dropout_lstm_cell,
    l2_regularized_loss,
    attention_unit,
    generate_attn_heatmap_summary,
)


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
                ids=features["context_x"],
                initializer=embeddings,
                scope="embedding_layer",
                reuse=True,
            )

            target_embeddings = tf.contrib.layers.embed_sequence(
                ids=features["target_x"],
                initializer=embeddings,
                scope="embedding_layer",
                reuse=True,
            )

            with tf.variable_scope("context_lstm"):
                context_hidden_states, _ = tf.nn.dynamic_rnn(
                    cell=dropout_lstm_cell(
                        hidden_units=params["hidden_units"],
                        initializer=params["initializer"],
                        keep_prob=params["keep_prob"],
                    ),
                    inputs=context_embeddings,
                    sequence_length=features["context_len"],
                    dtype=tf.float32,
                )
                c_avg = variable_len_batch_mean(
                    input_tensor=context_hidden_states,
                    seq_lengths=features["context_len"],
                    op_name="context_avg_pooling",
                )

            with tf.variable_scope("target_lstm"):
                target_hidden_states, _ = tf.nn.dynamic_rnn(
                    cell=dropout_lstm_cell(
                        hidden_units=params["hidden_units"],
                        initializer=params["initializer"],
                        keep_prob=params["keep_prob"],
                    ),
                    inputs=target_embeddings,
                    sequence_length=features["target_len"],
                    dtype=tf.float32,
                )
                t_avg = variable_len_batch_mean(
                    input_tensor=target_hidden_states,
                    seq_lengths=features["target_len"],
                    op_name="target_avg_pooling",
                )

            with tf.variable_scope("attention_layer", reuse=tf.AUTO_REUSE):
                c_r, ctxt_attn_info = attention_unit(
                    h_states=context_hidden_states,
                    hidden_units=params["hidden_units"],
                    seq_lengths=features["context_len"],
                    attn_focus=t_avg,
                    init=params["initializer"],
                    literal=features["context_lit"],
                )
                t_r, trg_attn_info = attention_unit(
                    h_states=target_hidden_states,
                    hidden_units=params["hidden_units"],
                    seq_lengths=features["target_len"],
                    attn_focus=c_avg,
                    init=params["initializer"],
                    literal=features["target_lit"],
                )

            generate_attn_heatmap_summary(trg_attn_info, ctxt_attn_info)

            final_sentence_rep = tf.concat([t_r, c_r], axis=1)

            logits = tf.layers.dense(
                inputs=final_sentence_rep,
                units=params["n_out_classes"],
                activation=tf.nn.tanh,
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

            loss = l2_regularized_loss(
                labels=labels, logits=logits, l2_weight=params["l2_weight"]
            )

            if mode == ModeKeys.EVAL:
                return EstimatorSpec(mode, loss=loss, predictions=predictions)

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
