import tensorflow as tf
from tensorflow.estimator import (  # pylint: disable=E0401
    EstimatorSpec,
    ModeKeys,
)
from tensorflow.contrib.rnn import (  # pylint: disable=E0611
    stack_bidirectional_dynamic_rnn
)
from tsaplay.models.Model import Model
from tsaplay.models.Chen2017.common import (
    params as default_params,
    ram_input_fn,
    ram_serving_fn,
    get_bounded_distance_vectors,
)
from tsaplay.utils._tf import (
    variable_len_batch_mean,
    attention_unit,
    dropout_lstm_cell,
    l2_regularized_loss,
    generate_attn_heatmap_summary,
    setup_embedding_layer,
    get_embedded_seq,
)


class RecurrentAttentionNetwork(Model):
    def _params(self):
        return default_params

    def _feature_columns(self):
        default = []
        return default

    def _train_input_fn(self):
        return lambda features, labels, batch_size: ram_input_fn(
            features, labels, batch_size
        )

    def _eval_input_fn(self):
        return lambda features, labels, batch_size: ram_input_fn(
            features, labels, batch_size, eval_input=True
        )

    def _serving_input_fn(self):
        return lambda features: ram_serving_fn(features)

    def _model_fn(self):
        def default(features, labels, mode, params=self.params):
            embedding_matrix = setup_embedding_layer(
                vocab_size=params["vocab_size"],
                dim_size=params["embedding_dim"],
                init=params["embedding_initializer"],
            )

            sentence_embeddings = get_embedded_seq(
                features["sentence_x"], embedding_matrix
            )
            target_embeddings = get_embedded_seq(
                features["target_x"], embedding_matrix
            )

            max_seq_len = tf.shape(sentence_embeddings)[1]

            target_avg = variable_len_batch_mean(
                input_tensor=target_embeddings,
                seq_lengths=features["target_len"],
                op_name="target_avg_pooling",
            )
            with tf.variable_scope("bi_lstm"):
                lstm_cell = dropout_lstm_cell(
                    hidden_units=params["hidden_units"],
                    initializer=params["initializer"],
                    keep_prob=params["keep_prob"],
                )
                memory_star, _, _ = stack_bidirectional_dynamic_rnn(
                    cells_fw=[lstm_cell] * params["n_lstm_layers"],
                    cells_bw=[lstm_cell] * params["n_lstm_layers"],
                    inputs=sentence_embeddings,
                    sequence_length=features["sentence_len"],
                    dtype=tf.float32,
                )

            distances = get_bounded_distance_vectors(
                left_bounds=features["target_left_bound"],
                right_bounds=features["target_right_bound"],
                seq_lens=features["sentence_len"],
                max_seq_len=max_seq_len,
            )

            seq_len_mask = tf.sequence_mask(
                features["sentence_len"], maxlen=max_seq_len, dtype=tf.float32
            )
            seq_len = tf.expand_dims(features["sentence_len"], axis=1)

            u_t = tf.cast(tf.divide(distances, seq_len), tf.float32)
            w_t = (1 - tf.abs(u_t)) * seq_len_mask

            u_t = tf.expand_dims(u_t, axis=2)
            w_t = tf.expand_dims(w_t, axis=2)

            memory = tf.concat([tf.multiply(memory_star, w_t), u_t], axis=2)

        return default
