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

params = {
    "batch_size": 25,
    "max_seq_length": 85,
    "n_out_classes": 3,
    "learning_rate": 0.1,
    "l2_weight": 1e-5,
    "momentum": 0.9,
    "keep_prob": 0.5,
    "hidden_units": 50,
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
        sentence_map, sentence_len, key="sentence"
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
