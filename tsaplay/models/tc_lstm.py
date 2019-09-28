import tensorflow as tf
from tsaplay.models.tsa_model import TsaModel
from tsaplay.utils.tf import sparse_reverse, variable_len_batch_mean, lstm_cell
from tsaplay.utils.debug import cprnt


class TcLstm(TsaModel):
    def set_params(self):
        return {
            ### Taken from https://github.com/jimmyyfeng/TD-LSTM/blob/master/tc_lstm.py ###
            "batch-size": 64,
            "hidden_units": 200,
            "epochs": 10,
            ###
            "learning_rate": 0.01,
            "initializer": tf.initializers.random_uniform(-0.003, 0.003),
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
            "target_ids": features["target_ids"],
        }

    def model_fn(self, features, labels, mode, params):
        max_left_len = tf.shape(features["left_emb"])[1]
        max_right_len = tf.shape(features["right_emb"])[1]

        with tf.name_scope("target_connection"):
            mean_target_embedding = variable_len_batch_mean(
                input_tensor=features["target_emb"],
                seq_lengths=features["target_len"],
                op_name="target_embedding_avg",
            )
            features["left_emb"] = tf.stack(
                values=[
                    features["left_emb"],
                    tf.ones(tf.shape(features["left_emb"]))
                    * mean_target_embedding,
                ],
                axis=2,
            )
            features["left_emb"] = tf.reshape(
                tensor=features["left_emb"],
                shape=[-1, max_left_len, 2 * params["_embedding_dim"]],
            )
            features["right_emb"] = tf.stack(
                values=[
                    features["right_emb"],
                    tf.ones(tf.shape(features["right_emb"]))
                    * mean_target_embedding,
                ],
                axis=2,
            )
            features["right_emb"] = tf.reshape(
                tensor=features["right_emb"],
                shape=[-1, max_right_len, 2 * params["_embedding_dim"]],
            )

        with tf.variable_scope("left_lstm"):
            _, final_states_left = tf.nn.dynamic_rnn(
                cell=lstm_cell(**params),
                inputs=features["left_emb"],
                sequence_length=features["left_len"],
                dtype=tf.float32,
            )

        with tf.variable_scope("right_lstm"):
            _, final_states_right = tf.nn.dynamic_rnn(
                cell=lstm_cell(**params),
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
