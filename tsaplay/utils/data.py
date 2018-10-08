import tensorflow as tf
from functools import wraps
from tsaplay.utils.io import cprnt
from tsaplay.utils.tf import sparse_sequences_to_dense, get_seq_lengths


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


def prep_dataset(tfrecord, params, processing_fn, mode):
    # shuffle_buffer = batch_size * 10
    # shuffle_buffer = 100000
    shuffle_buffer = params.get("shuffle-bufer", 100000)
    dataset = tf.data.TFRecordDataset(tfrecord)
    dataset = dataset.map(parse_tf_example)
    if processing_fn is not None:
        dataset = dataset.map(processing_fn)

    if mode == "EVAL":
        dataset = dataset.shuffle(buffer_size=shuffle_buffer)
    elif mode == "TRAIN":
        dataset = dataset.apply(
            tf.contrib.data.shuffle_and_repeat(buffer_size=shuffle_buffer)
        )

    dataset = dataset.batch(params["batch-size"])

    return dataset


def make_dense_features(features):
    dense_features = {}
    cprnt(features)
    for key in features:
        if "_ids" in key:
            name, _, _ = key.partition("_")
            name_str = sparse_sequences_to_dense(features[name])
            name_ids = sparse_sequences_to_dense(features[key])
            name_lens = get_seq_lengths(name_ids)
            dense_features.update(
                {
                    name: name_str,
                    name + "_ids_dense": name_ids,
                    name + "_len": name_lens,
                }
            )
    features.update(dense_features)
    return features
