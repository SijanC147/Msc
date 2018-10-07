import tensorflow as tf
from tensorflow.estimator import (  # pylint: disable=E0401
    EstimatorSpec,
    ModeKeys,
)
from tsaplay.models.TSAModel import TSAModel
from tsaplay.utils.tf import dropout_lstm_cell
from tsaplay.utils.decorators import prep_features


class Lstm(TSAModel):
    def set_params(self):
        return {
            "batch-size": 100,
            "n_out_classes": 3,
            "learning_rate": 0.01,
            "keep_prob": 0.8,
            "hidden_units": 100,
            "initializer": tf.initializers.random_uniform(
                minval=-0.03, maxval=0.03
            ),
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

    @prep_features(["sentence"])
    def model_fn(self, features, labels, mode, params):
        _, final_states = tf.nn.dynamic_rnn(
            cell=dropout_lstm_cell(
                hidden_units=params["hidden_units"],
                initializer=params["initializer"],
                keep_prob=params["keep_prob"],
            ),
            inputs=features["sentence_emb"],
            sequence_length=features["sentence_len"],
            dtype=tf.float32,
        )

        logits = tf.layers.dense(
            inputs=final_states.h, units=params["n_out_classes"]
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
