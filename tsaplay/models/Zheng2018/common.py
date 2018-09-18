import tensorflow as tf
from tensorflow.python.keras.preprocessing import (  # pylint: disable=E0611
    sequence
)
from tsaplay.utils._data import (
    make_labels_dataset_from_list,
    prep_features_for_dataset,
    wrap_mapping_length_literal,
    wrap_left_target_right_label,
    prep_dataset_and_get_iterator,
)

params = {
    "batch_size": 35,
    "max_seq_length": 85,
    "n_out_classes": 3,
    "learning_rate": 0.1,
    "l2_weight": 1e-5,
    "momentum": 0.9,
    "keep_prob": 0.5,
    "hidden_units": 100,
    "initializer": tf.initializers.random_uniform(minval=-0.1, maxval=0.1),
}


def lcr_rot_input_fn(
    features, labels, batch_size, max_seq_length, eval_input=False
):

    left_map, left_len = prep_features_for_dataset(
        mappings=features["mappings"]["left"], max_seq_length=max_seq_length
    )
    left = wrap_mapping_length_literal(left_map, left_len, features["left"])

    right_map, right_len = prep_features_for_dataset(
        mappings=features["mappings"]["right"], max_seq_length=max_seq_length
    )
    right = wrap_mapping_length_literal(
        right_map, right_len, features["right"]
    )

    target_map, target_len = prep_features_for_dataset(
        mappings=features["mappings"]["target"]
    )
    target = wrap_mapping_length_literal(
        target_map, target_len, features["target"]
    )

    labels = make_labels_dataset_from_list(labels)

    dataset = wrap_left_target_right_label(left, target, right, labels)

    iterator = prep_dataset_and_get_iterator(
        dataset=dataset,
        shuffle_buffer=len(features),
        batch_size=batch_size,
        eval_input=eval_input,
    )

    return iterator.get_next()
