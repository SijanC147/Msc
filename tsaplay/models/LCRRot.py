from os.path import join
import tensorflow as tf
from tensorflow.estimator import (  # pylint: disable=E0401
    EstimatorSpec,
    ModeKeys,
)
from tensorflow.contrib.rnn import (  # pylint: disable=E0611
    stack_bidirectional_dynamic_rnn
)
from tsaplay.models.TSAModel import TSAModel
from tsaplay.utils.tf import (
    variable_len_batch_mean,
    attention_unit,
    dropout_lstm_cell,
    l2_regularized_loss,
    generate_attn_heatmap_summary,
)
from tsaplay.utils.io import cprnt
from tsaplay.utils.decorators import addon
from tsaplay.models.addons import attn_heatmaps


class LCRRot(TSAModel):
    def set_params(self):
        return {
            "batch-size": 25,
            "n_out_classes": 3,
            "learning_rate": 0.1,
            "l2_weight": 1e-5,
            "momentum": 0.9,
            "keep_prob": 0.5,
            "hidden_units": 300,
            "initializer": tf.initializers.random_uniform(
                minval=-0.1, maxval=0.1
            ),
            "n_attn_heatmaps": 5,
        }

    @addon([attn_heatmaps])
    def model_fn(self, features, labels, mode, params):
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
                inputs=features["left_emb"],
                sequence_length=features["left_len"],
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
                inputs=features["right_emb"],
                sequence_length=features["right_len"],
                dtype=tf.float32,
            )

        with tf.variable_scope("left_t2c_attn"):
            r_l, left_attn_info = attention_unit(
                h_states=left_hidden_states,
                hidden_units=params["hidden_units"] * 2,
                seq_lengths=features["left_len"],
                attn_focus=r_t,
                init=params["initializer"],
                sp_literal=features["left"],
            )

        with tf.variable_scope("right_t2c_attn"):
            r_r, right_attn_info = attention_unit(
                h_states=right_hidden_states,
                hidden_units=params["hidden_units"] * 2,
                seq_lengths=features["right_len"],
                attn_focus=r_t,
                init=params["initializer"],
                sp_literal=features["right"],
            )

        with tf.variable_scope("left_c2t_attn"):
            r_t_l, left_target_attn_info = attention_unit(
                h_states=target_hidden_states,
                hidden_units=params["hidden_units"] * 2,
                seq_lengths=features["target_len"],
                attn_focus=tf.expand_dims(r_l, axis=1),
                init=params["initializer"],
                sp_literal=features["target"],
            )

        with tf.variable_scope("right_c2t_attn"):
            r_t_r, right_target_attn_info = attention_unit(
                h_states=target_hidden_states,
                hidden_units=params["hidden_units"] * 2,
                seq_lengths=features["target_len"],
                attn_focus=tf.expand_dims(r_r, axis=1),
                init=params["initializer"],
                sp_literal=features["target"],
            )

        generate_attn_heatmap_summary(
            left_attn_info,
            left_target_attn_info,
            right_target_attn_info,
            right_attn_info,
        )

        final_sentence_rep = tf.concat([r_l, r_t_l, r_t_r, r_r], axis=1)

        logits = tf.layers.dense(
            inputs=final_sentence_rep, units=params["n_out_classes"]
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
