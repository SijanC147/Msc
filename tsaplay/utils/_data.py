import tensorflow as tf
from tensorflow.python.keras.preprocessing import (  # pylint: disable=E0611
    sequence
)


def zip_str_join(first, second):
    return [" ".join([f.strip(), s.strip()]) for f, s in zip(first, second)]


def zip_list_join(first, second, reverse=False):
    if reverse:
        return [list(reversed(f + s)) for f, s in zip(first, second)]
    return [f + s for f, s in zip(first, second)]


def make_labels_dataset_from_list(labels):
    low_bound = min(labels)
    if low_bound < 0:
        labels = [label + abs(low_bound) for label in labels]
    return tf.data.Dataset.from_tensor_slices(labels)


def prep_features_for_dataset(mappings, max_seq_length=None):
    lengths = [len(mapping) for mapping in mappings]
    mappings = sequence.pad_sequences(
        sequences=mappings,
        maxlen=max_seq_length or max(lengths),
        truncating="post",
        padding="post",
        value=0,
    )
    return mappings, lengths


def wrap_mapping_length_literal(mapping, length, literal=None):
    if literal is None:
        dataset = tf.data.Dataset.from_tensor_slices((mapping, length))
        dataset = dataset.map(
            lambda mapping, length: {"x": mapping, "len": length}
        )
    else:
        dataset = tf.data.Dataset.from_tensor_slices(
            (mapping, length, literal)
        )
        dataset = dataset.map(
            lambda mapping, length, literal: {
                "x": mapping,
                "len": length,
                "lit": literal,
            }
        )
    return dataset


def wrap_left_target_right_label(left, target, right, label):
    dataset = tf.data.Dataset.zip((left, target, right, label))
    dataset = dataset.map(
        lambda left, target, right, label: (
            {"left": left, "target": target, "right": right},
            label,
        )
    )
    return dataset


def prep_dataset_and_get_iterator(
    dataset, shuffle_buffer, batch_size, eval_input
):
    if eval_input:
        dataset = dataset.shuffle(buffer_size=shuffle_buffer)
    else:
        dataset = dataset.apply(
            tf.contrib.data.shuffle_and_repeat(buffer_size=shuffle_buffer)
        )

    dataset = dataset.batch(batch_size=batch_size)

    iterator = dataset.make_one_shot_iterator()

    return iterator
