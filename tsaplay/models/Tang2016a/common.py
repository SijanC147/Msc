import tensorflow as tf
from tensorflow.python.keras.preprocessing import (  # pylint: disable=E0611
    sequence
)
from tsaplay.utils._data import (
    prep_dataset_and_get_iterator,
    zip_list_join,
    prep_features_for_dataset,
    wrap_mapping_length_literal,
    make_labels_dataset_from_list,
    wrap_left_target_right_label,
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
    sen_map, sen_len = prep_features_for_dataset(mappings=sentences)
    labels = make_labels_dataset_from_list(labels)

    dataset = tf.data.Dataset.from_tensor_slices((sen_map, sen_len, labels))
    dataset = dataset.map(
        lambda sentence, length, label: ({"x": sentence, "len": length}, label)
    )

    iterator = prep_dataset_and_get_iterator(
        dataset=dataset,
        shuffle_buffer=len(features),
        batch_size=batch_size,
        eval_input=eval_input,
    )

    return iterator.get_next()


def tdlstm_input_fn(
    features, labels, batch_size, max_seq_length, eval_input=False
):
    left_contexts = zip_list_join(
        features["mappings"]["left"], features["mappings"]["target"]
    )

    left_map, left_len = prep_features_for_dataset(
        mappings=left_contexts, max_seq_length=max_seq_length
    )
    left = wrap_mapping_length_literal(left_map, left_len)

    right_contexts = zip_list_join(
        features["mappings"]["target"],
        features["mappings"]["left"],
        reverse=True,
    )
    right_map, right_len = prep_features_for_dataset(
        mappings=right_contexts, max_seq_length=max_seq_length
    )
    right = wrap_mapping_length_literal(right_map, right_len)

    labels = make_labels_dataset_from_list(labels)

    dataset = tf.data.Dataset.zip((left, right, labels))
    dataset = dataset.map(
        lambda left, right, label: ({"left": left, "right": right}, label)
    )

    iterator = prep_dataset_and_get_iterator(
        dataset=dataset,
        shuffle_buffer=len(features),
        batch_size=batch_size,
        eval_input=eval_input,
    )

    return iterator.get_next()


def tclstm_input_fn(
    features, labels, batch_size, max_seq_length, eval_input=False
):
    left_contexts = zip_list_join(
        features["mappings"]["left"], features["mappings"]["target"]
    )

    left_map, left_len = prep_features_for_dataset(
        mappings=left_contexts, max_seq_length=max_seq_length
    )
    left = wrap_mapping_length_literal(left_map, left_len)

    right_contexts = zip_list_join(
        features["mappings"]["target"],
        features["mappings"]["left"],
        reverse=True,
    )
    right_map, right_len = prep_features_for_dataset(
        mappings=right_contexts, max_seq_length=max_seq_length
    )
    right = wrap_mapping_length_literal(right_map, right_len)

    target_map, target_len = prep_features_for_dataset(
        mappings=features["mappings"]["target"]
    )
    target = wrap_mapping_length_literal(target_map, target_len)

    labels = make_labels_dataset_from_list(labels)

    dataset = wrap_left_target_right_label(left, target, right, labels)

    iterator = prep_dataset_and_get_iterator(
        dataset=dataset,
        shuffle_buffer=len(features),
        batch_size=batch_size,
        eval_input=eval_input,
    )

    return iterator.get_next()
