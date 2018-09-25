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


def ian_input_fn(features, labels, batch_size, eval_input=False):
    context_literals = zip_str_join(features["left"], features["right"])
    context_mappings = zip_list_join(
        features["mappings"]["left"], features["mappings"]["right"]
    )

    contexts_map, contexts_len = pad_for_dataset(mappings=context_mappings)
    contexts = package_feature_dict(
        mappings=contexts_map,
        lengths=contexts_len,
        literals=context_literals,
        key="context",
    )

    target_map, target_len = pad_for_dataset(
        mappings=features["mappings"]["target"]
    )
    targets = package_feature_dict(
        mappings=target_map,
        lengths=target_len,
        literals=features["target"],
        key="target",
    )

    iterator = prep_dataset_and_get_iterator(
        features={**contexts, **targets},
        labels=labels,
        batch_size=batch_size,
        eval_input=eval_input,
    )

    return iterator.get_next()


def ian_serving_fn(features):
    return {
        "context_x": features["mappings"]["context"],
        "context_len": tf.add(
            features["lengths"]["left"], features["lengths"]["right"]
        ),
        "context_lit": tf.strings.join(
            [features["literals"]["left"], features["literals"]["right"]],
            separator=" ",
        ),
        "context_tok": tf.strings.join(
            [features["tok_enc"]["left"], features["tok_enc"]["right"]],
            separator="<SEP>",
        ),
        "target_x": features["mappings"]["target"],
        "target_len": features["lengths"]["target"],
        "target_lit": features["literals"]["target"],
        "target_tok": features["tok_enc"]["target"],
    }
