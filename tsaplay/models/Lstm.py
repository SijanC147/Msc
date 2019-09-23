import tensorflow as tf
from tsaplay.models.tsa_model import TsaModel
from tsaplay.utils.tf import lstm_cell


class Lstm(TsaModel):
    def set_params(self):
        return {
            ### Taken from https://github.com/jimmyyfeng/TD-LSTM/blob/master/lstm.py ###
            "batch-size": 100,  
            "hidden_units": 200,  
            "n_epoch": 10, # not implemented yet  
            ###
            "learning_rate": 0.01,
            "initializer": tf.initializers.random_uniform(-0.003, 0.003),
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

    def model_fn(self, features, labels, mode, params):
        _, final_states = tf.nn.dynamic_rnn(
            cell=lstm_cell(**params),
            inputs=features["sentence_emb"],
            sequence_length=features["sentence_len"],
            dtype=tf.float32,
        )

        logits = tf.layers.dense(
            inputs=final_states.h, units=params["_n_out_classes"]
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
