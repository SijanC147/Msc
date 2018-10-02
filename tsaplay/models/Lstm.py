import tensorflow as tf
from tensorflow.estimator import (  # pylint: disable=E0401
    EstimatorSpec,
    ModeKeys,
)
from tsaplay.models.TSAModel import TSAModel
from tsaplay.utils.tf import (
    dropout_lstm_cell,
    seq_lengths,
    sparse_sequences_to_dense,
)


class Lstm(TSAModel):
    def set_params(self):
        return {
            "batch_size": 100,
            "n_out_classes": 3,
            "learning_rate": 0.01,
            "keep_prob": 0.8,
            "hidden_units": 50,
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

    def model_fn(self, features, labels, mode, params):
        sentence_ids = sparse_sequences_to_dense(features["sentence_ids"])
        sentence_len = seq_lengths(sentence_ids)

        inputs = tf.contrib.layers.embed_sequence(
            ids=sentence_ids,
            vocab_size=params["vocab_size"],
            embed_dim=params["embedding_dim"],
            initializer=params["embedding_initializer"],
        )

        _, final_states = tf.nn.dynamic_rnn(
            cell=dropout_lstm_cell(
                hidden_units=params["hidden_units"],
                initializer=params["initializer"],
                keep_prob=params["keep_prob"],
            ),
            inputs=inputs,
            sequence_length=sentence_len,
            dtype=tf.float32,
        )

        logits = tf.layers.dense(
            inputs=final_states.h, units=params["n_out_classes"]
        )

        predictions = {
            "class_ids": tf.argmax(logits, 1),
            "probabilities": tf.nn.softmax(logits),
            "logits": logits,
        }

        if mode == ModeKeys.PREDICT:
            return EstimatorSpec(mode, predictions=predictions)

        loss = tf.losses.sparse_softmax_cross_entropy(
            labels=labels, logits=logits
        )

        if mode == ModeKeys.EVAL:
            return EstimatorSpec(mode, predictions=predictions, loss=loss)

        optimizer = tf.train.AdagradOptimizer(
            learning_rate=params["learning_rate"]
        )
        train_op = optimizer.minimize(
            loss, global_step=tf.train.get_global_step()
        )

        return EstimatorSpec(
            mode, loss=loss, train_op=train_op, predictions=predictions
        )
