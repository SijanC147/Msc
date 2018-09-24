import tensorflow as tf
from tensorflow.python.keras.preprocessing import (  # pylint: disable=E0611
    sequence
)
from tsaplay.utils._data import (
    zip_list_join,
    zip_str_join,
    pad_for_dataset,
    package_feature_dict,
    prep_dataset_and_get_iterator,
)
from tsaplay.utils._tf import masked_softmax

params = {
    "batch_size": 25,
    "n_out_classes": 3,
    "learning_rate": 0.1,
    "l2_weight": 1e-5,
    "keep_prob": 0.5,
    "lstm_hidden_units": 100,
    "gru_hidden_units": 50,
    "n_lstm_layers": 2,
    "n_hops": 3,
    "initializer": tf.initializers.random_uniform(minval=-0.1, maxval=0.1),
}


def ram_input_fn(features, labels, batch_size, eval_input=False):
    sentences = [
        l + t + r
        for l, t, r in zip(
            features["mappings"]["left"],
            features["mappings"]["target"],
            features["mappings"]["right"],
        )
    ]
    sentence_map, sentence_len = pad_for_dataset(sentences)
    sentences = package_feature_dict(
        sentence_map,
        sentence_len,
        literal=features["sentence"],
        key="sentence",
    )

    left_lens = [len(mapping) for mapping in features["mappings"]["left"]]
    right_lens = [len(mapping) for mapping in features["mappings"]["right"]]

    target_left_bounds = [left_len + 1 for left_len in left_lens]
    target_right_bounds = [
        sen_len - right_len
        for (sen_len, right_len) in zip(sentence_len, right_lens)
    ]
    target_map, target_len = pad_for_dataset(
        mappings=features["mappings"]["target"]
    )
    targets = package_feature_dict(
        target_map, target_len, literal=features["target"], key="target"
    )
    targets = {
        **targets,
        "target_left_bound": target_left_bounds,
        "target_right_bound": target_right_bounds,
    }

    iterator = prep_dataset_and_get_iterator(
        features={**sentences, **targets},
        labels=labels,
        batch_size=batch_size,
        eval_input=eval_input,
    )

    return iterator.get_next()


def ram_serving_fn(features):
    return {
        "sentence_x": features["mappings"]["sentence"],
        "sentence_len": features["lengths"]["sentence"],
        "sentence_lit": features["literals"]["sentence"],
        "target_x": features["mappings"]["target"],
        "target_len": features["lengths"]["target"],
        "target_lit": features["literals"]["target"],
    }


def get_bounded_distance_vectors(
    left_bounds, right_bounds, seq_lens, max_seq_len
):
    batch_size = tf.shape(seq_lens)[0]
    mask = tf.sequence_mask(lengths=seq_lens, maxlen=max_seq_len)

    seq_range = tf.range(start=1, limit=max_seq_len + 1)
    seq_range_tiled = tf.tile([seq_range], multiples=[batch_size, 1])

    left_bounds = tf.expand_dims(left_bounds, axis=1)
    right_bounds = tf.expand_dims(right_bounds, axis=1)
    left_bounds_tiled = tf.ones_like(seq_range_tiled) * left_bounds
    right_bounds_tiled = tf.ones_like(seq_range_tiled) * right_bounds

    distance_left = tf.where(
        tf.less_equal(seq_range_tiled, left_bounds_tiled),
        left_bounds_tiled,
        seq_range_tiled,
    )
    distance_left_right = tf.where(
        tf.greater(distance_left, right_bounds_tiled),
        right_bounds_tiled,
        distance_left,
    )

    distances = seq_range_tiled - distance_left_right

    distances_masked = tf.where(mask, distances, tf.zeros_like(distances))

    return distances_masked


def calculate_u_t_w_t(seq_lens, max_seq_len, distances):
    mask = tf.sequence_mask(seq_lens, maxlen=max_seq_len, dtype=tf.float32)

    seq_len = tf.expand_dims(seq_lens, axis=1)

    u_t = tf.divide(distances, seq_len)

    u_t = tf.cast(u_t, tf.float32)

    w_t = (1 - tf.abs(u_t)) * mask

    return u_t, w_t


def get_location_weighted_memory(memory_star, w_t, u_t):
    u_t = tf.expand_dims(u_t, axis=2)
    w_t = tf.expand_dims(w_t, axis=2)

    memory = tf.multiply(memory_star, w_t)
    memory = tf.concat([memory, u_t], axis=2)

    return memory


def var_len_concatenate(seq_lens, memory, v_target, prev_episode):
    max_seq_len = tf.shape(memory)[1]

    mask = tf.sequence_mask(seq_lens, maxlen=max_seq_len, dtype=tf.float32)
    mask = tf.expand_dims(mask, axis=2)

    prev_episode = tf.tile(prev_episode, multiples=[1, max_seq_len, 1])
    v_target = tf.tile(v_target, multiples=[1, max_seq_len, 1])

    mem_prev_e = tf.concat([memory, prev_episode * mask], axis=2)
    mem_prev_e_v_t = tf.concat([mem_prev_e, v_target * mask], axis=2)

    return mem_prev_e_v_t


def ram_attn_unit(seq_lens, attn_focus, weight_dim, init):
    batch_size = tf.shape(attn_focus)[0]
    max_seq_len = tf.shape(attn_focus)[1]
    w_att = tf.get_variable(
        name="weights",
        shape=[1, weight_dim],
        dtype=tf.float32,
        initializer=init,
    )
    b_att = tf.get_variable(
        name="bias", shape=[1], dtype=tf.float32, initializer=init
    )

    w_att_batch_dim = tf.expand_dims(w_att, axis=0)
    w_att_tiled = tf.tile(
        w_att_batch_dim, multiples=[batch_size * max_seq_len, 1, 1]
    )
    w_att_batched = tf.reshape(
        w_att_tiled, shape=[-1, max_seq_len, 1, weight_dim]
    )

    b_att_mask = tf.sequence_mask(
        seq_lens, maxlen=max_seq_len, dtype=tf.float32
    )
    b_att_masked = b_att_mask * b_att
    b_att_batched = tf.reshape(b_att_masked, shape=[batch_size, -1, 1, 1])

    attn_focus_transpose = tf.expand_dims(attn_focus, axis=3)
    g_score = tf.einsum("Baij,Bajk->Baik", w_att_batched, attn_focus_transpose)
    g_score = g_score + b_att_batched

    g_score = tf.squeeze(g_score, axis=3)

    softmax_mask = tf.cast(b_att_mask, dtype=tf.bool)

    attn_weights = masked_softmax(logits=g_score, mask=softmax_mask)

    return attn_weights

