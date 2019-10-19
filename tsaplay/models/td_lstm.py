# pylint: disable=line-too-long
import tensorflow as tf
from tsaplay.models.tsa_model import TsaModel
from tsaplay.utils.tf import sparse_reverse, lstm_cell
from tsaplay.utils.addons import addon, early_stopping


class TdLstm(TsaModel):
    def set_params(self):
        return {
            # * From original paper
            "learning_rate": 0.01,
            "initializer": tf.initializers.random_uniform(-0.003, 0.003),
            # ? Suggestions from https://github.com/jimmyyfeng/TD-LSTM/blob/master/td_lstm.py
            "hidden_units": 200,
            # ? Using same batch size to compare LSTM, TDLSTM and TDLSTM
            "batch-size": 64,
            # ? Following approach of Moore et al. 2018, using early stopping
            "epochs": 300,
            "early_stopping_patience": 10,
            "early_stopping_minimum_iter": 30,
            "early_stopping_metric": "loss",
        }

    @classmethod
    def processing_fn(cls, features):
        return {
            "left_ids": tf.sparse_concat(
                sp_inputs=[features["left_ids"], features["target_ids"]],
                axis=1,
            ),
            "right_ids": sparse_reverse(
                tf.sparse_concat(
                    sp_inputs=[features["right_ids"], features["target_ids"]],
                    axis=1,
                )
            ),
        }

    @addon([early_stopping])
    def model_fn(self, features, labels, mode, params):
        with tf.variable_scope("left_lstm"):
            _, final_states_left = tf.nn.dynamic_rnn(
                cell=lstm_cell(**params, mode=mode),
                inputs=features["left_emb"],
                sequence_length=features["left_len"],
                dtype=tf.float32,
            )

        with tf.variable_scope("right_lstm"):
            _, final_states_right = tf.nn.dynamic_rnn(
                cell=lstm_cell(**params, mode=mode),
                inputs=features["right_emb"],
                sequence_length=features["right_len"],
                dtype=tf.float32,
            )

        concatenated_final_states = tf.concat(
            [final_states_left.h, final_states_right.h], axis=1
        )

        logits = tf.layers.dense(
            inputs=concatenated_final_states,
            units=params["_n_out_classes"],
            kernel_initializer=params["initializer"],
            bias_initializer=params["initializer"],
        )

        loss = tf.losses.sparse_softmax_cross_entropy(
            labels=labels, logits=logits
        )

        optimizer = tf.train.AdagradOptimizer(
            learning_rate=params["learning_rate"]
        )

        return self.make_estimator_spec(
            mode=mode, logits=logits, optimizer=optimizer, loss=loss
        )
