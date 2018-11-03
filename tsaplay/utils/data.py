from collections import defaultdict, Iterable
from itertools import chain, groupby
from operator import itemgetter
from inspect import getsource
from hashlib import md5
from functools import partial, wraps
import numpy as np
import spacy
from spacy.attrs import ORTH  # pylint: disable=E0611
from tqdm import tqdm
from tsaplay.constants import RANDOM_SEED, SPACY_MODEL
from tsaplay.utils.filters import default_token_filter


def wrap_parsing_fn(parsing_fn):
    @wraps(parsing_fn)
    def wrapper(path):
        try:
            sentences, targets, offsets, labels = parsing_fn(path)
            return {
                "sentences": sentences,
                "targets": targets,
                "offsets": offsets,
                "labels": labels,
            }
        except ValueError:
            sentences, targets, labels = parsing_fn(path)
            offsets = [
                sentence.lower().find(target.lower())
                for sentence, target in zip(sentences, targets)
            ]
            return {
                "sentences": sentences,
                "targets": targets,
                "offsets": offsets,
                "labels": labels,
            }

    return wrapper


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


def merge_corpora(*corpi):
    corpi_items = [map(tuple, corpus.items()) for corpus in corpi]
    corpi_items = sorted(chain(*corpi_items))
    corpus = [
        (key, sum(j for _, j in group))
        for key, group in groupby(corpi_items, key=itemgetter(0))
    ]
    corpus.sort(key=itemgetter(1), reverse=True)
    return {word: count for word, count in corpus}


def corpora_vocab(*corpora):
    all_vocab = [list(map(str.lower, [*corpus])) for corpus in corpora]
    return list(set(sum(all_vocab, [])))


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


def stringify(list_element):
    if isinstance(list_element, str):
        return list_element
    if isinstance(list_element, partial):
        return str(list_element.keywords)
    if callable(list_element):
        return getsource(list_element)
    if hasattr(list_element, "sort"):
        list_element = list(set(map(str.lower, list_element)))
        list_element.sort()
    return str(list_element)


def hash_data(data):
    if not data:
        return ""
    if not isinstance(data, Iterable):
        data = str(data)
    if isinstance(data, str):
        data = data.encode()
    if isinstance(data, bytes):
        return md5(data).hexdigest()
    data = map(stringify, data)
    data = map(str.encode, data)
    data = [md5(el).hexdigest() for el in data]
    data.sort()
    return md5(str(data).encode()).hexdigest()


def tokenize_data(include=None, **data_dicts):
    include = set(map(str.lower, include)) if include else None
    docs = [
        sum(
            partition_sentences(
                data_dict["sentences"],
                data_dict["targets"],
                data_dict.get("offsets"),
            ),
            [],
        )
        for data_dict in data_dicts.values()
    ]
    doc_lengths = [len(doc) for doc in docs]
    docs = sum(docs, [])
    nlp = spacy.load(SPACY_MODEL, disable=["parser", "ner"])
    doc_pipe = nlp.pipe(docs, batch_size=100, n_threads=-1)
    tokens = [
        [
            token.text.lower()
            for token in filter(default_token_filter, doc)
            if token.text.lower() in include or not include
        ]
        for doc in tqdm(doc_pipe, total=len(docs), desc="Tokenizing Data")
    ]
    all_tokens_zipped = zip(
        [*data_dicts], split_list(tokens, counts=doc_lengths)
    )
    parts = ["Left", "Target", "Right"]

    return {
        key: {
            part: tokens
            for part, tokens in zip(parts, split_list(all_tokens, parts=3))
        }
        for key, all_tokens in all_tokens_zipped
    }

