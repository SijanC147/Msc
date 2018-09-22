import tensorflow as tf
from tensorflow.python.keras.preprocessing import (  # pylint: disable=E0611
    sequence
)
from tsaplay.utils._data import (
    prep_features_for_dataset,
    package_feature_dict,
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
    left = package_feature_dict(
        left_map, left_len, literal=features["left"], key="left"
    )

    right_map, right_len = prep_features_for_dataset(
        mappings=features["mappings"]["right"], max_seq_length=max_seq_length
    )
    right = package_feature_dict(
        right_map, right_len, literal=features["right"], key="right"
    )

    target_map, target_len = prep_features_for_dataset(
        mappings=features["mappings"]["target"]
    )
    target = package_feature_dict(
        target_map, target_len, literal=features["target"], key="target"
    )

    iterator = prep_dataset_and_get_iterator(
        features={**left, **target, **right},
        labels=labels,
        batch_size=batch_size,
        eval_input=eval_input,
    )

    return iterator.get_next()
