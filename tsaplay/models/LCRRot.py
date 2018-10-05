import tensorflow as tf
from os.path import join
from tensorflow.contrib.layers import embed_sequence  # pylint: disable=E0611
from tensorflow.estimator import (  # pylint: disable=E0401
    EstimatorSpec,
    ModeKeys,
)
from tensorflow.contrib.rnn import (  # pylint: disable=E0611
    stack_bidirectional_dynamic_rnn
)
from tsaplay.models.TSAModel import TSAModel
from tsaplay.utils.tf import (
    sparse_sequences_to_dense,
    seq_lengths,
    variable_len_batch_mean,
    attention_unit,
    dropout_lstm_cell,
    l2_regularized_loss,
    generate_attn_heatmap_summary,
    setup_embedding_layer,
    get_embedded_seq,
)
from tsaplay.utils.io import cprnt
from tsaplay.utils.decorators import addon, inputs
from tsaplay.models.addons import attn_heatmaps


class LCRRot(TSAModel):
    def set_params(self):
        return {
            "batch_size": 50,
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
    @inputs(["left", "target", "right"])
    def model_fn(self, features, labels, mode, params):
        # left_ids = sparse_sequences_to_dense(features["left_ids"])
        # target_ids = sparse_sequences_to_dense(features["target_ids"])
        # right_ids = sparse_sequences_to_dense(features["right_ids"])

        # with tf.variable_scope("embedding_layer", reuse=True):
        #     embeddings = tf.get_variable("embeddings")

        # left_embedded = tf.nn.embedding_lookup(embeddings, left_ids)
        # target_embedded = tf.nn.embedding_lookup(embeddings, target_ids)
        # right_embedded = tf.nn.embedding_lookup(embeddings, right_ids)

        # left_len = seq_lengths(left_ids)
        # target_len = seq_lengths(target_ids)
        # right_len = seq_lengths(right_ids)

        left_embedded = features["left_emb"]
        target_embedded = features["target_emb"]
        right_embedded = features["right_emb"]

        left_len = features["left_len"]
        target_len = features["target_len"]
        right_len = features["right_len"]

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
                inputs=target_embedded,
                sequence_length=target_len,
                dtype=tf.float32,
            )
            r_t = variable_len_batch_mean(
                input_tensor=target_hidden_states,
                seq_lengths=target_len,
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
                inputs=left_embedded,
                sequence_length=left_len,
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
                inputs=right_embedded,
                sequence_length=right_len,
                dtype=tf.float32,
            )

        with tf.variable_scope("left_t2c_attn"):
            r_l, left_attn_info = attention_unit(
                h_states=left_hidden_states,
                hidden_units=params["hidden_units"] * 2,
                seq_lengths=left_len,
                attn_focus=r_t,
                init=params["initializer"],
                sp_literal=features["left"],
            )

        with tf.variable_scope("right_t2c_attn"):
            r_r, right_attn_info = attention_unit(
                h_states=right_hidden_states,
                hidden_units=params["hidden_units"] * 2,
                seq_lengths=right_len,
                attn_focus=r_t,
                init=params["initializer"],
                sp_literal=features["right"],
            )

        with tf.variable_scope("left_c2t_attn"):
            r_t_l, left_target_attn_info = attention_unit(
                h_states=target_hidden_states,
                hidden_units=params["hidden_units"] * 2,
                seq_lengths=target_len,
                attn_focus=tf.expand_dims(r_l, axis=1),
                init=params["initializer"],
                sp_literal=features["target"],
            )

        with tf.variable_scope("right_c2t_attn"):
            r_t_r, right_target_attn_info = attention_unit(
                h_states=target_hidden_states,
                hidden_units=params["hidden_units"] * 2,
                seq_lengths=target_len,
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
            return EstimatorSpec(mode, predictions=predictions, loss=loss)

        optimizer = tf.train.MomentumOptimizer(
            learning_rate=params["learning_rate"], momentum=params["momentum"]
        )
        train_op = optimizer.minimize(
            loss, global_step=tf.train.get_global_step()
        )

        return EstimatorSpec(
            mode, loss=loss, train_op=train_op, predictions=predictions
        )

