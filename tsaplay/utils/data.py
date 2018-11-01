from collections import defaultdict
from itertools import chain, groupby
from operator import itemgetter
import numpy as np
import tensorflow as tf
import spacy
from spacy.attrs import ORTH  # pylint: disable=E0611
from tqdm import tqdm
from tsaplay.constants import RANDOM_SEED, SPACY_MODEL
from tsaplay.utils.filters import default_token_filter
from tsaplay.utils.tf import sparse_sequences_to_dense, get_seq_lengths
from tsaplay.utils.io import cprnt


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
            if features.get(name):
                dense_features.update(
                    {name: sparse_sequences_to_dense(features[name])}
                )
            name_ids = sparse_sequences_to_dense(features[key])
            name_lens = get_seq_lengths(name_ids)
            dense_features.update(
                {name + "_ids": name_ids, name + "_len": name_lens}
            )
    features.update(dense_features)
    return features


def prep_dataset(tfrecords, params, processing_fn, mode):
    shuffle_buffer = params.get("shuffle-buffer", 30)
    parallel_calls = params.get("parallel_calls", 4)
    parallel_batches = params.get("parallel_batches", parallel_calls)
    prefetch_buffer = params.get("prefetch_buffer", 100)
    dataset = tf.data.Dataset.list_files(file_pattern=tfrecords)
    dataset = dataset.apply(
        tf.contrib.data.parallel_interleave(
            tf.data.TFRecordDataset,
            cycle_length=3,
            buffer_output_elements=prefetch_buffer,
            prefetch_input_elements=parallel_calls,
        )
    )
    if mode == "EVAL":
        dataset = dataset.shuffle(buffer_size=shuffle_buffer)
    elif mode == "TRAIN":
        dataset = dataset.apply(
            tf.contrib.data.shuffle_and_repeat(buffer_size=shuffle_buffer)
        )

    dataset = dataset.apply(
        tf.contrib.data.map_and_batch(
            lambda example: processing_fn(*parse_tf_example(example)),
            params["batch-size"],
            num_parallel_batches=parallel_batches,
        )
    )
    dataset = dataset.map(
        lambda features, labels: (make_dense_features(features), labels),
        num_parallel_calls=parallel_calls,
    )
    dataset = dataset.cache()

    return dataset


def class_dist_info(labels, all_classes=None):
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

    np.random.seed(RANDOM_SEED)
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


def merge_dicts(*dicts):
    new_dict = defaultdict(list)
    for key, value in chain.from_iterable(map(dict.items, dicts)):
        new_dict[key] += value
    return dict(new_dict)


def class_dist_stats(classes=None, **data_dicts):
    stats = {}
    for key, value in data_dicts.items():
        stats[key] = stats.get(key, {})
        dist_data = class_dist_info(value["labels"], all_classes=classes)
        for (_class, count, dist) in zip(*dist_data):
            stats[key].update(
                {str(_class): {"count": str(count), "percent": str(dist)}}
            )
    return stats


def target_offsets(sentences, targets):
    return [
        sentence.lower().find(target.lower())
        for (sentence, target) in zip(sentences, targets)
    ]


def partition_sentences(sentences, targets, offsets=None):
    offsets = offsets or target_offsets(sentences, targets)
    left_ctxts = [sen[:off] for (sen, off) in zip(sentences, offsets)]
    targets = list(map(str.strip, targets))
    right_off = [off + len(trg) for (off, trg) in zip(offsets, targets)]
    right_ctxts = [sen[r_off:] for (sen, r_off) in zip(sentences, right_off)]
    left_ctxts = list(map(str.strip, left_ctxts))
    right_ctxts = list(map(str.strip, right_ctxts))
    return left_ctxts, targets, right_ctxts


def zero_norm_labels(labels):
    return [label + abs(min(labels)) for label in labels]


def split_list(data, counts=None, parts=None):
    counts = counts or ([int(len(data) / parts)] * parts)
    offsets = [0] + np.cumsum(counts).tolist()
    return [data[offsets[i] : offsets[i + 1]] for i in range(len(offsets) - 1)]


def generate_corpus(docs, mode=None):
    desc = (
        "Building {mode} Corpus".format(mode=mode.capitalize())
        if mode
        else "Building Corpus"
    )
    nlp = spacy.load(SPACY_MODEL, disable=["parser", "ner"])
    doc_pipe = nlp.pipe(set(docs), batch_size=100, n_threads=-1)
    word_counts = (
        [
            (nlp.vocab.strings[key], count)
            for (key, count) in doc.count_by(ORTH).items()
        ]
        for doc in tqdm(doc_pipe, total=len(docs), desc=desc)
    )
    word_counts = sorted(chain(*word_counts))
    words = [
        (key, sum(j for _, j in group))
        for key, group in groupby(word_counts, key=itemgetter(0))
    ]
    words.sort(key=itemgetter(1), reverse=True)
    return {word: count for word, count in words}
