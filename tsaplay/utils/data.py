from contextlib import redirect_stdout
from io import StringIO
from re import search
from collections import defaultdict, Iterable
from itertools import chain, groupby
from operator import itemgetter
from inspect import getsource
from hashlib import md5
from functools import partial, wraps
from tqdm import tqdm
import numpy as np
from tsaplay.constants import NP_RANDOM_SEED, DELIMITER
from tsaplay.utils.filters import (
    default_token_filter,
    group_filter_fns,
    filters_registry,
)
from tsaplay.utils.spacy import pipe_docs, word_counts, pipe_vocab


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
    target_dists = (
        target_dists / 100 if np.sum(target_dists) == 100 else target_dists
    )
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

    np.random.seed(NP_RANDOM_SEED)
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


def accumulate_dicts(*args, accum_fn=None, default=None, **kwargs):
    dicts = list(args or []) + (list(kwargs.values()) if kwargs else [])
    new_dict = defaultdict(default or list)
    accum_fn = accum_fn if callable(accum_fn) else (lambda prev, curr: prev + curr)
    for key, value in chain.from_iterable(map(dict.items, dicts)):
        try:
            new_dict[key] = accum_fn(new_dict[key], value)
        except TypeError:
            new_dict[key] = accum_fn(new_dict[key], default(value))
    return dict(new_dict)


def merge_corpora(*corpora):
    corpi_items = [map(tuple, corpus.items()) for corpus in corpora]
    corpi_items = sorted(chain(*corpi_items))
    corpus = [
        (key, sum(j for _, j in group))
        for key, group in groupby(corpi_items, key=itemgetter(0))
    ]
    corpus.sort(key=itemgetter(1), reverse=True)
    return {word: count for word, count in corpus}


def corpora_vocab(*corpora, case_insensitive=None):
    all_vocab = [
        list(map(str.lower, [*corpus]) if case_insensitive else [*corpus])
        for corpus in corpora
    ]
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


def vocab_case_insensitive(vocab):
    return not (True in [bool(search("[A-Z]", word)) for word in vocab])


def accum_tuple_list_gen(gen, sort=True):
    data = sorted(chain(*gen))
    data = [
        (key, sum(j for _, j in group))
        for key, group in groupby(data, key=itemgetter(0))
    ]
    if sort:
        data.sort(key=itemgetter(1), reverse=True)
    return data


def generate_corpus(docs, mode=None):
    desc = (
        "Building {mode} Corpus".format(mode=mode.capitalize())
        if mode
        else "Building Corpus"
    )
    words = word_counts(set(docs), pbar_desc=desc)
    counts = accum_tuple_list_gen(words)
    return {word: count for word, count in counts}


def stringify(list_element):
    if isinstance(list_element, str):
        return list_element
    if isinstance(list_element, partial):
        keywords = list_element.keywords
        keywords = {key: val for key, val in sorted(keywords.items())}
        return str(
            {
                "function": list_element.func.__qualname__,
                "args": str(list_element.args),
                "kwargs": keywords,
            }
        )
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
        data = data.encode(encoding="utf-8")
    if isinstance(data, bytes):
        return md5(data).hexdigest()
    data = list(map(stringify, data))
    data = list(map(lambda d: d.encode(encoding="utf-8"), data))
    data = [md5(el).hexdigest() for el in data]
    data.sort()
    return md5(str(data).encode(encoding="utf-8")).hexdigest()


def tokenize_data(include=None, case_insensitive=None, **data_dicts):
    include = (
        set(map(str.lower, include) if case_insensitive else include)
        if include
        else []
    )
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
    tokens = [
        [
            (token.text.lower() if case_insensitive else token.text)
            for token in filter(default_token_filter, doc)
            if (
                (token.text.lower() if case_insensitive else token.text)
                in include
            )
            or not include
        ]
        for doc in pipe_docs(docs, pbar_desc="Tokenizing data")
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


def filter_vocab_list(vocab, filters, case_insensitive=None, incl_report=None):
    filtered = vocab
    filter_report = None
    orig_len = len(filtered)
    filters = [
        filters_registry(f)
        if isinstance(f, str) and filters_registry(f)
        else f
        for f in filters
    ]
    filter_sets = list(filter(lambda filt: not callable(filt), filters))
    if filter_sets:
        filter_sets = (
            list(map(str.lower, sum(map(list, filter_sets), [])))
            if case_insensitive
            else sum(map(list, filter_sets), [])
        )
        filtered = list(set(filtered) & set(filter_sets))

    filter_fns = list(filter(callable, filters))
    if filter_fns:
        filter_fn, req_pipes, report_header = group_filter_fns(
            *filter_fns, stdout_report=incl_report
        )
        vocab_pipe = pipe_vocab(
            filtered, pipes=req_pipes, pbar_desc="Parsing vocabulary"
        )
        if incl_report:
            report_str = StringIO()
            with redirect_stdout(report_str):
                filtered = [
                    [token.text for token in filter(filter_fn, doc)]
                    for doc in vocab_pipe
                ]
            filter_report = report_str.getvalue().split("\n")
            filter_report = [row.split(DELIMITER) for row in filter_report]
            filter_report = [report_header] + filter_report
        else:
            filtered = [
                [token.text for token in filter(filter_fn, doc)]
                for doc in vocab_pipe
            ]
        filtered = sum(tqdm(filtered, desc="Filtering vocabulary"), [])

    filtered_length = len(filtered)
    reduction = ((orig_len - filtered_length) / orig_len) * 100
    filter_details = {
        "vocab": {
            "original": orig_len,
            "filtered": filtered_length,
            "reduction": reduction,
        },
        "filters": {
            "functions": list(map(stringify, filter_fns)),
            "sets": list(map(stringify, filter_sets)),
        },
    }

    return filtered, filter_report, filter_details


def tokens_by_assigned_id(words, ids, start=None, stop=None, keys=None):
    ids_dict = defaultdict(set)
    try:
        words = sum(words, [])
        ids = sum(ids, [])
    except TypeError:
        pass
    start = start or min(ids)
    stop = stop or (start + len(keys)) if keys else max(ids)
    for word, index in zip(words, ids):
        if start <= index <= stop:
            key = keys[index - start] if keys else str(index)
            ids_dict[key] |= (
                set([str(word, "utf-8")])
                if isinstance(word, bytes)
                else set([word])
            )
    return {key: list(value) for key, value in ids_dict.items()}

