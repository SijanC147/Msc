import tensorflow as tf
from tensorflow.python.keras.preprocessing import (  # pylint: disable=E0611
    sequence
)
from tsaplay.utils._data import (
    prep_dataset_and_get_iterator,
    zip_list_join,
    pad_for_dataset,
    package_feature_dict,
    make_labels_dataset_from_list,
    tf_encoded_tokenisation,
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


def lstm_input_fn(features, labels, batch_size, eval_input=False):
    sentences = [
        l + t + r
        for l, t, r in zip(
            features["mappings"]["left"],
            features["mappings"]["target"],
            features["mappings"]["right"],
        )
    ]
    sen_map, sen_len = pad_for_dataset(sentences)
    sentence = package_feature_dict(
        mappings=sen_map, lengths=sen_len, literals=features["sentence"]
    )

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
        "lit": features["literal"]["sentence"],
        "tok": features["tok_enc"]["sentence"],
    }


def tdlstm_input_fn(features, labels, batch_size, eval_input=False):
    left_contexts = zip_list_join(
        features["mappings"]["left"], features["mappings"]["target"]
    )

    left_map, left_len = pad_for_dataset(left_contexts)
    left = package_feature_dict(
        mappings=left_map,
        lengths=left_len,
        literals=features["left"],
        key="left",
    )

    right_contexts = zip_list_join(
        features["mappings"]["target"],
        features["mappings"]["left"],
        reverse=True,
    )
    right_map, right_len = pad_for_dataset(right_contexts)
    right = package_feature_dict(
        mappings=right_map,
        lengths=right_len,
        literals=features["right"],
        key="right",
    )

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
        "left_tok": features["tok_enc"]["left"],
        "right_x": features["mappings"]["target_right"],
        "right_len": tf.add(
            features["lengths"]["target"], features["lengths"]["right"]
        ),
        "right_tok": features["tok_enc"]["right"],
    }


def tclstm_input_fn(features, labels, batch_size, eval_input=False):
    left_contexts = zip_list_join(
        features["mappings"]["left"], features["mappings"]["target"]
    )
    left_map, left_len = pad_for_dataset(left_contexts)
    left = package_feature_dict(
        mappings=left_map,
        lengths=left_len,
        literals=features["left"],
        key="left",
    )

    right_contexts = zip_list_join(
        features["mappings"]["target"],
        features["mappings"]["left"],
        reverse=True,
    )
    right_map, right_len = pad_for_dataset(right_contexts)
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


def tclstm_serving_fn(features):
    return {
        "left_x": features["mappings"]["left_target"],
        "left_len": tf.add(
            features["lengths"]["left"], features["lengths"]["target"]
        ),
        "left_tok": features["tok_enc"]["left"],
        "right_x": features["mappings"]["target_right"],
        "right_len": tf.add(
            features["lengths"]["target"], features["lengths"]["right"]
        ),
        "right_tok": features["tok_enc"]["right"],
        "target_x": features["mappings"]["target"],
        "target_len": features["lengths"]["target"],
        "target_tok": features["tok_enc"]["target"],
    }
