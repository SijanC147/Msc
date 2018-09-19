import tensorflow as tf
from tensorflow.python.keras.preprocessing import (  # pylint: disable=E0611
    sequence
)
from tsaplay.utils._data import (
    zip_list_join,
    zip_str_join,
    prep_features_for_dataset,
    wrap_mapping_length_literal,
    make_labels_dataset_from_list,
    wrap_left_target_right_label,
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


def ian_input_fn(
    features, labels, batch_size, max_seq_length, eval_input=False
):
    context_literals = zip_str_join(features["left"], features["right"])
    context_mappings = zip_list_join(
        features["mappings"]["left"], features["mappings"]["right"]
    )

    contexts_map, contexts_len = prep_features_for_dataset(
        mappings=context_mappings, max_seq_length=max_seq_length
    )
    contexts = wrap_mapping_length_literal(
        contexts_map, contexts_len, context_literals
    )

    target_map, target_len = prep_features_for_dataset(
        mappings=features["mappings"]["target"]
    )
    targets = wrap_mapping_length_literal(
        target_map, target_len, features["target"]
    )

    labels = make_labels_dataset_from_list(labels)

    dataset = tf.data.Dataset.zip((contexts, targets, labels))

    dataset = dataset.map(
        lambda context, target, label: (
            {"context": context, "target": target},
            label,
        )
    )

    iterator = prep_dataset_and_get_iterator(
        dataset=dataset,
        shuffle_buffer=len(features),
        batch_size=batch_size,
        eval_input=eval_input,
    )

    return iterator.get_next()
