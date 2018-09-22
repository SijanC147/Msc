import tensorflow as tf
from tensorflow.python.keras.preprocessing import (  # pylint: disable=E0611
    sequence
)
from tsaplay.utils._data import (
    prep_dataset_and_get_iterator,
    zip_list_join,
    prep_features_for_dataset,
    package_feature_dict,
    make_labels_dataset_from_list,
)

params = {
    "batch_size": 25,
    "max_seq_length": 140,
    "n_out_classes": 3,
    "learning_rate": 0.01,
    "keep_prob": 0.8,
    "hidden_units": 200,
    "initializer": tf.initializers.random_uniform(minval=-0.03, maxval=0.03),
}


def lstm_input_fn(
    features, labels, batch_size, max_seq_length, eval_input=False
):
    sentences = [
        l + t + r
        for l, t, r in zip(
            features["mappings"]["left"],
            features["mappings"]["target"],
            features["mappings"]["right"],
        )
    ]
    sen_map, sen_len = prep_features_for_dataset(
        mappings=sentences, max_seq_length=max_seq_length
    )
    sentence = package_feature_dict(sen_map, sen_len)

    iterator = prep_dataset_and_get_iterator(
        features=sentence,
        labels=labels,
        batch_size=batch_size,
        eval_input=eval_input,
    )

    return iterator.get_next()


def lstm_serving_fn(features):
    return {
        "x": features["mappings"]["sentence"],
        "len": features["lengths"]["sentence"],
    }


def tdlstm_input_fn(
    features, labels, batch_size, max_seq_length, eval_input=False
):
    left_contexts = zip_list_join(
        features["mappings"]["left"], features["mappings"]["target"]
    )

    left_map, left_len = prep_features_for_dataset(
        mappings=left_contexts, max_seq_length=max_seq_length
    )
    left = package_feature_dict(left_map, left_len, key="left")

    right_contexts = zip_list_join(
        features["mappings"]["target"],
        features["mappings"]["left"],
        reverse=True,
    )
    right_map, right_len = prep_features_for_dataset(
        mappings=right_contexts, max_seq_length=max_seq_length
    )
    right = package_feature_dict(right_map, right_len, key="right")

    iterator = prep_dataset_and_get_iterator(
        features={**left, **right},
        labels=labels,
        batch_size=batch_size,
        eval_input=eval_input,
    )

    return iterator.get_next()


def tdlstm_serving_fn(features):
    return {
        "left_x": features["mappings"]["left_target"],
        "left_len": tf.add(
            features["lengths"]["left"], features["lengths"]["target"]
        ),
        "right_x": features["mappings"]["target_right"],
        "right_len": tf.add(
            features["lengths"]["target"], features["lengths"]["right"]
        ),
    }


def tclstm_input_fn(
    features, labels, batch_size, max_seq_length, eval_input=False
):
    left_contexts = zip_list_join(
        features["mappings"]["left"], features["mappings"]["target"]
    )
    left_map, left_len = prep_features_for_dataset(
        mappings=left_contexts, max_seq_length=max_seq_length
    )
    left = package_feature_dict(left_map, left_len, key="left")

    right_contexts = zip_list_join(
        features["mappings"]["target"],
        features["mappings"]["left"],
        reverse=True,
    )
    right_map, right_len = prep_features_for_dataset(
        mappings=right_contexts, max_seq_length=max_seq_length
    )
    right = package_feature_dict(right_map, right_len, key="right")

    target_map, target_len = prep_features_for_dataset(
        mappings=features["mappings"]["target"]
    )
    target = package_feature_dict(target_map, target_len, key="target")

    iterator = prep_dataset_and_get_iterator(
        features={**left, **target, **right},
        labels=labels,
        batch_size=batch_size,
        eval_input=eval_input,
    )

    return iterator.get_next()


def tclstm_serving_fn(features):
    return {
        "left_x": features["mappings"]["left_target"],
        "left_len": tf.add(
            features["lengths"]["left"], features["lengths"]["target"]
        ),
        "right_x": features["mappings"]["target_right"],
        "right_len": tf.add(
            features["lengths"]["target"], features["lengths"]["right"]
        ),
        "target_x": features["mappings"]["target"],
        "target_len": features["lengths"]["target"],
    }
