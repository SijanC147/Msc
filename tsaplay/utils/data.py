import numpy as np
import tensorflow as tf
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


def make_dense_features(features):
    dense_features = {}
    for key in features:
        if "_ids" in key:
            name, _, _ = key.partition("_")
            name_str = sparse_sequences_to_dense(features[name])
            name_ids = sparse_sequences_to_dense(features[key])
            name_lens = get_seq_lengths(name_ids)
            dense_features.update(
                {
                    name: name_str,
                    name + "_ids": name_ids,
                    name + "_len": name_lens,
                }
            )
    features.update(dense_features)
    return features


def prep_dataset(tfrecords, params, processing_fn, mode):
    shuffle_buffer = params.get("shuffle-bufer", 50)
    dataset = tf.data.Dataset.list_files(file_pattern=tfrecords)
    dataset = dataset.interleave(
        tf.data.TFRecordDataset, cycle_length=5, block_length=1
    )
    if mode == "EVAL":
        dataset = dataset.shuffle(buffer_size=shuffle_buffer)
    elif mode == "TRAIN":
        dataset = dataset.apply(
            tf.contrib.data.shuffle_and_repeat(buffer_size=shuffle_buffer)
        )

    def parse_and_process(example):
        return processing_fn(*parse_tf_example(example))

    dataset = dataset.apply(
        tf.contrib.data.map_and_batch(
            parse_and_process, params["batch-size"], num_parallel_batches=5
        )
    )
    dataset = dataset.map(
        lambda features, labels: (make_dense_features(features), labels)
    )
    dataset = dataset.prefetch(buffer_size=None)

    return dataset


def get_class_distribution(labels, all_classes=None):
    classes, counts = np.unique(labels, return_counts=True)
    if all_classes is not None and len(classes) != len(all_classes):
        all_counts = []
        counts_list = counts.tolist()
        classes_list = classes.tolist()
        for _class in all_classes:
            try:
                all_counts.append(counts_list[classes_list.index(_class)])
            except ValueError:
                all_counts.append(0)
        classes = np.array(all_classes)
        counts = np.array(all_counts)
    total = np.sum(counts)
    dists = np.round(np.divide(counts, total) * 100).astype(np.int32)
    return classes, counts, dists


def re_distribute_counts(labels, target_dists):
    target_dists = np.array(target_dists)
    unique, counts = np.unique(labels, return_counts=True)

    if len(counts) != len(target_dists):
        raise ValueError(
            "Expected {0} distribution values, got {1}".format(
                len(unique), len(target_dists)
            )
        )

    if 1 in target_dists:
        return np.where(target_dists == 1, counts, 0)

    valid_counts = counts * target_dists
    counts = np.where(valid_counts != 0, counts, np.inf)

    while not np.isinf(min(counts)):
        lowest_valid_count = np.where(counts == min(counts), counts, 0)
        totals = np.where(
            lowest_valid_count != 0,
            np.floor_divide(lowest_valid_count, target_dists),
            np.inf,
        )
        total = np.min(totals)
        candidate_counts = np.floor(total * target_dists)
        counts = np.where(candidate_counts > counts, counts, np.Inf)

    target_counts = candidate_counts.astype(int)
    return unique, target_counts


def resample_data_dict(data_dict, target_dists):
    labels = [label for label in data_dict["labels"]]
    classes, target_counts = re_distribute_counts(labels, target_dists)
    numpy_dtype = np.dtype(
        [
            (key, (type(value[0]), max([len(v) for v in value])))
            if isinstance(value[0], str)
            else (key, type(value[0]))
            for key, value in data_dict.items()
        ]
    )

    labels_index = [*data_dict].index("labels")
    samples = list(zip(*data_dict.values()))
    samples_by_class = {}
    for _class in classes:
        samples_by_class[str(_class)] = np.asarray(
            [s for s in samples if s[labels_index] == _class], numpy_dtype
        )

    resampled = np.concatenate(
        [
            np.random.choice(
                samples_by_class[str(_class)], count, replace=False
            )
            for _class, count in zip(classes, target_counts)
        ],
        axis=0,
    )
    np.random.shuffle(resampled)

    resampled = resampled.tolist()

    resampled_data_dict = {}
    for index, value in enumerate(data_dict):
        resampled_data_dict[value] = [sample[index] for sample in resampled]

    return resampled_data_dict
