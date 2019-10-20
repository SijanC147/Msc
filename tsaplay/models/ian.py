# pylint: disable=line-too-long
import tensorflow as tf
from tsaplay.models.tsa_model import TsaModel
from tensorflow.estimator import ModeKeys  # noqa
from tsaplay.utils.tf import (
    variable_len_batch_mean,
    lstm_cell,
    l2_regularized_loss,
    attention_unit,
    generate_attn_heatmap_summary,
)
from tsaplay.utils.addons import addon, attn_heatmaps, early_stopping


class Ian(TsaModel):
    def set_params(self):
        return {
            # * From original paper
            "hidden_units": 300,
            "l2_weight": 1e-5,
            "keep_prob": 0.5,
            "initializer": tf.initializers.random_uniform(-0.1, 0.1),
            "bias_initializer": tf.initializers.zeros(),
            "lstm_initial_bias": 0,
            # ! Not quoted paper, using value from LCRROT paper
            "momentum": 0.9,
            # ? Suggestions from https://github.com/songyouwei/ABSA-PyTorch/blob/master/train.py
            "learning_rate": 0.1,
            "batch-size": 64,
            # "epochs": 30,
            "early_stopping_minimum_iter": 30,
            # ? Following approach of Moore et al. 2018, using early stopping
            "epochs": 300,
            "early_stopping_patience": 10,
            # "early_stopping_metric": "macro-f1",
            "early_stopping_metric": "loss",
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

    @addon([attn_heatmaps, early_stopping])
    def model_fn(self, features, labels, mode, params):
        with tf.variable_scope("context_lstm"):
            features["context_emb"] = tf.nn.dropout(
                features["context_emb"], keep_prob=params["keep_prob"]
            )
            context_hidden_states, _ = tf.nn.dynamic_rnn(
                cell=lstm_cell(**params),
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
            features["target_emb"] = tf.nn.dropout(
                features["target_emb"], keep_prob=params["keep_prob"]
            )
            target_hidden_states, _ = tf.nn.dynamic_rnn(
                cell=lstm_cell(**params),
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
            context_hidden_states = tf.nn.dropout(
                context_hidden_states, keep_prob=params["keep_prob"]
            )
            c_r, ctxt_attn_info = attention_unit(
                h_states=context_hidden_states,
                hidden_units=params["hidden_units"],
                seq_lengths=features["context_len"],
                attn_focus=t_avg,
                init=params["initializer"],
                bias_init=params["bias_initializer"],
                sp_literal=features["context"],
            )

            target_hidden_states = tf.nn.dropout(
                target_hidden_states, keep_prob=params["keep_prob"]
            )
            t_r, trg_attn_info = attention_unit(
                h_states=target_hidden_states,
                hidden_units=params["hidden_units"],
                seq_lengths=features["target_len"],
                attn_focus=c_avg,
                init=params["initializer"],
                bias_init=params["bias_initializer"],
                sp_literal=features["target"],
            )

        generate_attn_heatmap_summary(trg_attn_info, ctxt_attn_info)

        final_sentence_rep = tf.concat([t_r, c_r], axis=1)
        final_sentence_rep = tf.nn.dropout(
            final_sentence_rep, keep_prob=params["keep_prob"]
        )

        logits = tf.layers.dense(
            inputs=final_sentence_rep,
            units=params["_n_out_classes"],
            activation=tf.nn.tanh,
            kernel_initializer=params["initializer"],
            bias_initializer=params.get(
                "bias_initializer", params["initializer"]
            ),
        )

        loss = l2_regularized_loss(
            labels=labels, logits=logits, l2_weight=params["l2_weight"]
        )

        optimizer = tf.train.MomentumOptimizer(
            learning_rate=params["learning_rate"], momentum=params["momentum"]
        )

        return self.make_estimator_spec(
            mode=mode, logits=logits, optimizer=optimizer, loss=loss
        )
