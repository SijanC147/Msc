# pylint: disable=line-too-long
import tensorflow as tf
from tsaplay.models.tsa_model import TsaModel
from tsaplay.utils.tf import lstm_cell
from tsaplay.utils.addons import addon, early_stopping


class Lstm(TsaModel):
    def set_params(self):
        return {
            # * From original paper
            "learning_rate": 0.01,
            "initializer": tf.initializers.random_uniform(-0.003, 0.003),
            # ? Suggestions from https://github.com/jimmyyfeng/TD-LSTM/blob/master/lstm.py
            "hidden_units": 200,
            # ? Using same batch size to compare LSTM, TDLSTM and TDLSTM
            "batch-size": 64,
            # ? Following approach of Moore et al. 2018, using early stopping
            "epochs": 300,
            "early_stopping_patience": 10,
            "early_stopping_minimum_iter": 30,
            "early_stopping_metric": "macro-f1",
        }

    @classmethod
    def processing_fn(cls, features):
        return {
            "sentence_ids": tf.sparse_concat(
                sp_inputs=[
                    features["left_ids"],
                    features["target_ids"],
                    features["right_ids"],
                ],
                axis=1,
            )
        }

    @addon([early_stopping])
    def model_fn(self, features, labels, mode, params):
        _, final_states = tf.nn.dynamic_rnn(
            cell=lstm_cell(**params, mode=mode),
            inputs=features["sentence_emb"],
            sequence_length=features["sentence_len"],
            dtype=tf.float32,
        )

        logits = tf.layers.dense(
            inputs=final_states.h,
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
