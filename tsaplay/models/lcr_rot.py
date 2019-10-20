# pylint: disable=line-too-long
import tensorflow as tf
from tensorflow.contrib.rnn import (  # pylint: disable=E0611
    stack_bidirectional_dynamic_rnn,
)
from tsaplay.models.tsa_model import TsaModel
from tsaplay.utils.tf import (
    variable_len_batch_mean,
    attention_unit,
    lstm_cell,
    l2_regularized_loss,
    generate_attn_heatmap_summary,
)
from tsaplay.utils.addons import addon, attn_heatmaps, early_stopping


class LcrRot(TsaModel):
    def set_params(self):
        return {
            # * From original paper
            "learning_rate": 0.1,
            "keep_prob": 0.5,
            "hidden_units": 300,
            "l2_weight": 1e-5,
            "momentum": 0.9,
            "initializer": tf.initializers.random_uniform(-0.1, 0.1),
            "bias_initializer": tf.initializers.zeros(),
            "lstm_initial_bias": 0,
            # ? Suggestions from https://github.com/NUSTM/ABSC/tree/master/models/ABSC_Zozoz
            "batch-size": 25,
            # "epochs": 50,
            "early_stopping_minimum_iter": 50,
            # ? Following approach of Moore et al. 2018, using early stopping
            "epochs": 300,
            "early_stopping_patience": 10,
            # "early_stopping_metric": "macro-f1",
            "early_stopping_metric": "loss",
        }

    @addon([attn_heatmaps, early_stopping])
    def model_fn(self, features, labels, mode, params):
        with tf.variable_scope("target_bi_lstm"):
            features["target_emb"] = tf.nn.dropout(
                features["target_emb"], keep_prob=params["keep_prob"]
            )
            target_hidden_states, _, _ = stack_bidirectional_dynamic_rnn(
                cells_fw=[lstm_cell(**params, mode=mode)],
                cells_bw=[lstm_cell(**params, mode=mode)],
                inputs=features["target_emb"],
                sequence_length=features["target_len"],
                dtype=tf.float32,
            )
            r_t = variable_len_batch_mean(
                input_tensor=target_hidden_states,
                seq_lengths=features["target_len"],
                op_name="target_avg_pooling",
            )

        with tf.variable_scope("left_bi_lstm"):
            features["left_emb"] = tf.nn.dropout(
                features["left_emb"], keep_prob=params["keep_prob"]
            )
            left_hidden_states, _, _ = stack_bidirectional_dynamic_rnn(
                cells_fw=[lstm_cell(**params, mode=mode)],
                cells_bw=[lstm_cell(**params, mode=mode)],
                inputs=features["left_emb"],
                sequence_length=features["left_len"],
                dtype=tf.float32,
            )

        with tf.variable_scope("right_bi_lstm"):
            features["right_emb"] = tf.nn.dropout(
                features["right_emb"], keep_prob=params["keep_prob"]
            )
            right_hidden_states, _, _ = stack_bidirectional_dynamic_rnn(
                cells_fw=[lstm_cell(**params, mode=mode)],
                cells_bw=[lstm_cell(**params, mode=mode)],
                inputs=features["right_emb"],
                sequence_length=features["right_len"],
                dtype=tf.float32,
            )

        with tf.variable_scope("left_t2c_attn"):
            left_hidden_states = tf.nn.dropout(
                left_hidden_states, keep_prob=params["keep_prob"]
            )
            r_l, left_attn_info = attention_unit(
                h_states=left_hidden_states,
                hidden_units=params["hidden_units"] * 2,
                seq_lengths=features["left_len"],
                attn_focus=r_t,
                init=params["initializer"],
                bias_init=params["bias_initializer"],
                sp_literal=features["left"],
            )

        with tf.variable_scope("right_t2c_attn"):
            right_hidden_states = tf.nn.dropout(
                right_hidden_states, keep_prob=params["keep_prob"]
            )
            r_r, right_attn_info = attention_unit(
                h_states=right_hidden_states,
                hidden_units=params["hidden_units"] * 2,
                seq_lengths=features["right_len"],
                attn_focus=r_t,
                init=params["initializer"],
                bias_init=params["bias_initializer"],
                sp_literal=features["right"],
            )

        target_hidden_states = tf.nn.dropout(
            target_hidden_states, keep_prob=params["keep_prob"]
        )

        with tf.variable_scope("left_c2t_attn"):
            r_t_l, left_target_attn_info = attention_unit(
                h_states=target_hidden_states,
                hidden_units=params["hidden_units"] * 2,
                seq_lengths=features["target_len"],
                attn_focus=tf.expand_dims(r_l, axis=1),
                init=params["initializer"],
                bias_init=params["bias_initializer"],
                sp_literal=features["target"],
            )

        with tf.variable_scope("right_c2t_attn"):
            r_t_r, right_target_attn_info = attention_unit(
                h_states=target_hidden_states,
                hidden_units=params["hidden_units"] * 2,
                seq_lengths=features["target_len"],
                attn_focus=tf.expand_dims(r_r, axis=1),
                init=params["initializer"],
                bias_init=params["bias_initializer"],
                sp_literal=features["target"],
            )

        generate_attn_heatmap_summary(
            left_attn_info,
            left_target_attn_info,
            right_target_attn_info,
            right_attn_info,
        )

        final_sentence_rep = tf.concat([r_l, r_t_l, r_t_r, r_r], axis=1)

        final_sentence_rep = tf.nn.dropout(
            final_sentence_rep, keep_prob=params["keep_prob"]
        )
        logits = tf.layers.dense(
            inputs=final_sentence_rep,
            units=params["_n_out_classes"],
            kernel_initializer=params["initializer"],
            bias_initializer=params["bias_initializer"],
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
