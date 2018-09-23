import tensorflow as tf
from itertools import chain
from os import makedirs
from os.path import join, exists
from collections import defaultdict
from tensorflow.python.keras.preprocessing import (  # pylint: disable=E0611
    sequence
)
from tsaplay.datasets.Dataset import Dataset, DATASETS
from tsaplay.utils._io import pickle_file


def zip_str_join(first, second):
    return [" ".join([f.strip(), s.strip()]) for f, s in zip(first, second)]


def zip_list_join(first, second, reverse=False):
    if reverse:
        return [list(reversed(f + s)) for f, s in zip(first, second)]
    return [f + s for f, s in zip(first, second)]


def concat_dicts_lists(first, second):
    new_dict = defaultdict(list)
    for k, v in chain(first.items(), second.items()):
        new_dict[k] = new_dict[k] + v

    return dict(new_dict)


def bundle_datasets(*datasets, rebuild=False):
    dataset_names = []
    train_dict = {}
    test_dict = {}
    for dataset in datasets:
        if isinstance(dataset, Dataset) and dataset.name not in dataset_names:
            dataset_names.append(dataset.name)
            train_dict = concat_dicts_lists(dataset.train_dict, train_dict)
            test_dict = concat_dicts_lists(dataset.test_dict, test_dict)

    dataset_name = "_".join(dataset_names)
    gen_path = join(DATASETS.PARENT_DIR, "_generated", dataset_name)

    makedirs(gen_path, exist_ok=True)

    train_dict_path = join(gen_path, "train_dict.pkl")
    test_dict_path = join(gen_path, "test_dict.pkl")

    if not exists(train_dict_path) and not rebuild:
        pickle_file(train_dict_path, train_dict)

    if not exists(test_dict_path) and not rebuild:
        pickle_file(test_dict_path, test_dict)

    return Dataset(path=gen_path, parser=None)


def make_labels_dataset_from_list(labels):
    low_bound = min(labels)
    if low_bound < 0:
        labels = [label + abs(low_bound) for label in labels]
    return tf.data.Dataset.from_tensor_slices(labels)


def pad_for_dataset(mappings):
    lengths = [len(mapping) for mapping in mappings]
    mappings = sequence.pad_sequences(
        sequences=mappings,
        maxlen=max(lengths) + 1,
        truncating="post",
        padding="post",
        value=0,
    )
    return mappings, lengths


def package_feature_dict(mapping, length, key=None, literal=None):
    if key is None:
        pre = ""
    else:
        pre = key + "_"

    if literal is None:
        return {pre + "x": mapping, pre + "len": length}
    else:
        return {pre + "x": mapping, pre + "len": length, pre + "lit": literal}


def prep_dataset_and_get_iterator(features, labels, batch_size, eval_input):
    shuffle_buffer = len(labels)

    features = tf.data.Dataset.from_tensor_slices(features)
    labels = make_labels_dataset_from_list(labels)

    dataset = tf.data.Dataset.zip((features, labels))

    if eval_input:
        dataset = dataset.shuffle(buffer_size=shuffle_buffer)
    else:
        dataset = dataset.apply(
            tf.contrib.data.shuffle_and_repeat(buffer_size=shuffle_buffer)
        )

    dataset = dataset.batch(batch_size=batch_size)

    iterator = dataset.make_one_shot_iterator()

    return iterator
