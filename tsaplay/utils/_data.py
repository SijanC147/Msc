import tensorflow as tf
from itertools import chain
from os import makedirs
from os.path import join, exists
from collections import defaultdict
from tsaplay.datasets.Dataset import Dataset, DATASETS
from tsaplay.utils._io import pickle_file
from tsaplay.utils._nlp import tokenize_phrase


def parse_tf_example(example):
    feature_spec = {
        "left": tf.VarLenFeature(dtype=tf.string),
        "target": tf.VarLenFeature(dtype=tf.string),
        "right": tf.VarLenFeature(dtype=tf.string),
        "left_ids": tf.VarLenFeature(dtype=tf.int64),
        "target_ids": tf.VarLenFeature(dtype=tf.int64),
        "right_ids": tf.VarLenFeature(dtype=tf.int64),
        "labels": tf.FixedLenFeature(dtype=tf.int64, shape=[]),
    }
    parsed_example = tf.parse_example([example], features=feature_spec)

    features = {
        "left": parsed_example["left"],
        "target": parsed_example["target"],
        "right": parsed_example["right"],
        "left_ids": parsed_example["left_ids"],
        "target_ids": parsed_example["target_ids"],
        "right_ids": parsed_example["right_ids"],
    }
    labels = tf.squeeze(parsed_example["labels"], axis=0)

    return (features, labels)


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
    gen_path = join(DATASETS.DATA_DIR, "_generated", dataset_name)

    makedirs(gen_path, exist_ok=True)

    train_dict_path = join(gen_path, "train_dict.pkl")
    test_dict_path = join(gen_path, "test_dict.pkl")

    if not exists(train_dict_path) and not rebuild:
        pickle_file(train_dict_path, train_dict)

    if not exists(test_dict_path) and not rebuild:
        pickle_file(test_dict_path, test_dict)

    return Dataset(path=gen_path, parser=None)

