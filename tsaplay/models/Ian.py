import tensorflow as tf
from tensorflow.estimator import (  # pylint: disable=E0401
    EstimatorSpec,
    ModeKeys,
)
from tsaplay.models.TSAModel import TSAModel
from tsaplay.utils.decorators import addon, prep_features
from tsaplay.utils.tf import (
    variable_len_batch_mean,
    dropout_lstm_cell,
    l2_regularized_loss,
    attention_unit,
    generate_attn_heatmap_summary,
)
from tsaplay.models.addons import attn_heatmaps


class Ian(TSAModel):
    def set_params(self):
        return {
            "batch_size": 25,
            "max_seq_length": 85,
            "n_out_classes": 3,
            "learning_rate": 0.1,
            "l2_weight": 1e-5,
            "momentum": 0.9,
            "keep_prob": 0.5,
            "hidden_units": 50,
            "initializer": tf.initializers.random_uniform(
                minval=-0.1, maxval=0.1
            ),
        }

    @classmethod
    def processing_fn(cls, features):
        return {
            "context": tf.sparse_concat(
                sp_inputs=[features["left"], features["right"]], axis=1
            ),
            "context_ids": tf.sparse_concat(
                sp_inputs=[features["left_ids"], features["right_ids"]], axis=1
            ),
            "target": features["target"],
            "target_ids": features["target_ids"],
        }

    @addon([attn_heatmaps])
    @prep_features(["context", "target"])
    def model_fn(self, features, labels, mode, params):
        with tf.variable_scope("context_lstm"):
            context_hidden_states, _ = tf.nn.dynamic_rnn(
                cell=dropout_lstm_cell(
                    hidden_units=params["hidden_units"],
                    initializer=params["initializer"],
                    keep_prob=params["keep_prob"],
                ),
                inputs=features["context_emb"],
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
                inputs=features["target_emb"],
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
                sp_literal=features["context"],
            )
            t_r, trg_attn_info = attention_unit(
                h_states=target_hidden_states,
                hidden_units=params["hidden_units"],
                seq_lengths=features["target_len"],
                attn_focus=c_avg,
                init=params["initializer"],
                sp_literal=features["target"],
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
            learning_rate=params["learning_rate"], momentum=params["momentum"]
        )
        train_op = optimizer.minimize(
            loss, global_step=tf.train.get_global_step()
        )

        return EstimatorSpec(
            mode, loss=loss, train_op=train_op, predictions=predictions
        )
