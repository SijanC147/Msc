import tensorflow as tf
from tensorflow.python.keras.preprocessing import (  # pylint: disable=E0611
    sequence
)
from tsaplay.utils._data import (
    pad_for_dataset,
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


def lcr_rot_input_fn(features, labels, batch_size, eval_input=False):

    left_map, left_len = pad_for_dataset(features["mappings"]["left"])
    left = package_feature_dict(
        mappings=left_map,
        lengths=left_len,
        literals=features["left"],
        key="left",
    )

    right_map, right_len = pad_for_dataset(features["mappings"]["right"])
    right = package_feature_dict(
        mappings=right_map,
        lengths=right_len,
        literals=features["right"],
        key="right",
    )

    target_map, target_len = pad_for_dataset(features["mappings"]["target"])
    target = package_feature_dict(
        mappings=target_map,
        lengths=target_len,
        literals=features["target"],
        key="target",
    )

    iterator = prep_dataset_and_get_iterator(
        features={**left, **target, **right},
        labels=labels,
        batch_size=batch_size,
        eval_input=eval_input,
    )

    return iterator.get_next()


def lcr_rot_serving_fn(features):
    return {
        "left_x": features["mappings"]["left"],
        "left_len": features["lengths"]["left"],
        "left_lit": features["literals"]["left"],
        "left_tok": features["tok_enc"]["left"],
        "target_x": features["mappings"]["target"],
        "target_len": features["lengths"]["target"],
        "target_lit": features["literals"]["target"],
        "target_tok": features["tok_enc"]["target"],
        "right_x": features["mappings"]["right"],
        "right_len": features["lengths"]["right"],
        "right_lit": features["literals"]["right"],
        "right_tok": features["tok_enc"]["right"],
    }
