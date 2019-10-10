from os.path import join, exists, dirname
from os import makedirs, walk
from functools import partial
from warnings import warn
from ast import literal_eval
from math import ceil
from copy import deepcopy
import re
import numpy as np

from tsaplay.constants import (
    FEATURES_DATA_PATH,
    DEFAULT_OOV_FN,
    BUCKET_TOKEN,
    TF_RANDOM_SEED,
    ASSETS_PATH,
)
from tsaplay.utils.tf import (
    ids_lookup_table,
    fetch_lookup_ops,
    run_lookups,
    make_tf_examples,
    partitioner_num_shards,
    embedding_initializer_fn,
)
from tsaplay.utils.io import (
    cprnt,
    pickle_file,
    unpickle_file,
    write_tfrecords,
    write_vocab_file,
    read_vocab_file,
    dump_json,
    search_dir,
)
from tsaplay.utils.data import (
    accumulate_dicts,
    merge_corpora,
    corpora_vocab,
    split_list,
    hash_data,
    tokenize_data,
    stringify,
    tokens_by_assigned_id,
    vocab_case_insensitive,
    lower_corpus,
)


class FeatureProvider:
    def __init__(self, datasets, embedding, **kwargs):
        self._oov_train_threshold = kwargs.get("oov_train", 1)
        self._num_oov_buckets = max(kwargs.get("oov_buckets", 1), 1)
        self._oov_fn = self._resolve_oov_fn(kwargs.get("oov_fn"))

        self._datasets = datasets
        self._embedding = embedding

        self._name = None
        self._uid = None
        self._gen_dir = None
        self._vocab = None
        self._class_labels = None
        self._train_dict = None
        self._test_dict = None
        self._train_corpus = None
        self._test_corpus = None
        self._train_oov_vocab = None
        self._test_oov_vocab = None
        self._train_tokens = None
        self._test_tokens = None
        # self._train_tfrecords = None
        # self._test_tfrecords = None
        self._oov_buckets = None

        self._init_uid()
        self._init_gen_dir()
        for mode in ["train", "test"]:
            self._init_data_dict(mode)
            self._init_corpus(mode)
        self._init_vocab()
        self._init_token_data()
        self._init_embedding_params()
        self._init_tfrecords()
        self._write_info_file()

    @property
    def name(self):
        return self._name

    @property
    def uid(self):
        return self._uid

    @property
    def datasets(self):
        try:
            return list(self._datasets)
        except TypeError:
            return [self._datasets]

    @property
    def embedding(self):
        return self._embedding

    @property
    def gen_dir(self):
        return self._gen_dir

    @property
    def class_labels(self):
        return self._class_labels

    @property
    def train_tfrecords(self):
        return join(self.gen_dir, "_train", "*.tfrecord")

    @property
    def test_tfrecords(self):
        return join(self.gen_dir, "_test", "*.tfrecord")

    @property
    def embedding_params(self):
        return self._embedding_params

    def steps_per_epoch(self, batch_size):
        train_samples = self._train_dict.get("labels")
        if not train_samples:
            raise ValueError(
                "Cannot get epoch of unintialized Feature Provider"
            )
        return ceil(len(train_samples) / batch_size), len(train_samples)

    def _init_uid(self):
        datasets_uids = [dataset.uid for dataset in self.datasets]
        dataset_names = [dataset.name for dataset in self.datasets]
        oov_policy = [self._oov_train_threshold, self._num_oov_buckets]
        uid_data = [self._embedding.uid] + datasets_uids + oov_policy
        cprnt(
            INFO="""INFO Feature Data:
Embedding: {embedding_uid}
Dataset(s): {datasets_uids}
Train OOV Freq: {train_oov_threshold}
OOV Buckets: {oov_buckets}
OOV Init Fn: {function} \t args: {args} \t kwargs: {kwargs}
""".format_map(
                {
                    "embedding_uid": self._embedding.uid,
                    "datasets_uids": "\t".join(datasets_uids),
                    "train_oov_threshold": self._oov_train_threshold,
                    "oov_buckets": self._num_oov_buckets,
                    **literal_eval(stringify(self._oov_fn)),
                }
            )
        )
        self._uid = hash_data(uid_data)
        name_data = [self._embedding.name] + dataset_names + [self._uid]
        self._name = "--".join(name_data)

    def _init_gen_dir(self):
        data_root = FEATURES_DATA_PATH
        gen_dir = join(data_root, self._name)
        if not exists(gen_dir):
            makedirs(gen_dir)
        self._gen_dir = gen_dir

    def _init_data_dict(self, mode):
        data_dict_attr = "_{mode}_dict".format(mode=mode)
        data_dict_file = "_{mode}_dict.pkl".format(mode=mode)
        data_dict_path = join(self._gen_dir, data_dict_file)
        if exists(data_dict_path):
            data_dict = unpickle_file(data_dict_path)
        else:
            data_dicts = (
                getattr(dataset, data_dict_attr) for dataset in self.datasets
            )
            data_dict = accumulate_dicts(*data_dicts)
            pickle_file(path=data_dict_path, data=data_dict)
        class_labels = self._class_labels or []
        class_labels = set(class_labels + data_dict["labels"])
        self._class_labels = list(class_labels)
        setattr(self, data_dict_attr, data_dict)

    def _init_corpus(self, mode):
        corpus_attr = "_{mode}_corpus".format(mode=mode)
        corpus_file = "_{mode}_corpus.pkl".format(mode=mode)
        corpus_path = join(self._gen_dir, corpus_file)
        if exists(corpus_path):
            corpus = unpickle_file(corpus_path)
        else:
            corpora = (
                lower_corpus(getattr(dataset, corpus_attr))
                if self._embedding.case_insensitive
                else getattr(dataset, corpus_attr)
                for dataset in self.datasets
            )
            corpus = merge_corpora(*corpora)
            pickle_file(path=corpus_path, data=corpus)
        setattr(self, corpus_attr, corpus)

    def _init_vocab(self):
        vocab_file_templ = "_vocab{ext}"
        vocab_file = vocab_file_templ.format(ext=".txt")
        vocab_file_path = join(self._gen_dir, vocab_file)
        # train_oov_file_path = join(self._gen_dir, "_train_oov.txt")
        self._vocab_file = vocab_file_path
        # if not exists(self._vocab_file) or not exists(train_oov_file_path):
        if not exists(self._vocab_file):
            self._vocab = deepcopy(self._embedding.vocab)
            #! include training vocabulary terms above the specified
            #! occurrance count if 0, all training vocab will be assigned
            #! buckets
            if self._oov_train_threshold > 0:
                train_vocab = set(
                    corpora_vocab(
                        self._train_corpus,
                        threshold=self._oov_train_threshold
                        # {
                        #     word: count
                        #     for word, count in self._train_corpus.items()
                        #     if count >= self._oov_train_threshold
                        # },
                        # case_insensitive=self._embedding.case_insensitive,
                    )
                )
                train_oov_vocab = list(train_vocab - set(self._vocab))
                train_oov_vocab.sort()
                self._vocab += train_oov_vocab
            write_vocab_file(vocab_file_path, self._vocab)
            # write_vocab_file(train_oov_file_path, self._train_oov_vocab)
        else:
            self._vocab = read_vocab_file(vocab_file_path)
            # self._train_oov_vocab = read_vocab_file(train_oov_file_path)
        vocab_tsv_file = vocab_file_templ.format(ext=".tsv")
        vocab_tsv_path = join(self._gen_dir, vocab_tsv_file)
        if not exists(vocab_tsv_path):
            oov_buckets_tokens = [
                BUCKET_TOKEN.format(num=n + 1)
                for n in range(self._num_oov_buckets)
            ]
            vocab_tsv = self._vocab + oov_buckets_tokens
            write_vocab_file(vocab_tsv_path, vocab_tsv)

    def _init_token_data(self):
        to_tokenize = {}
        for mode in ["train", "test"]:
            token_data_attr = "_{mode}_tokens".format(mode=mode)
            token_data_file = "_{mode}_tokens.pkl".format(mode=mode)
            token_data_path = join(self._gen_dir, token_data_file)
            if exists(token_data_path):
                token_data = unpickle_file(token_data_path)
                setattr(self, token_data_attr, token_data)
            else:
                data_dict_attr = "_{mode}_dict".format(mode=mode)
                data_dict = getattr(self, data_dict_attr)
                to_tokenize[mode] = data_dict
        if to_tokenize:
            #! Regardless of buckets, all vocab must be tokenized,
            #! otherwise risk experiment failing with empty target
            include = set(self._vocab) | set(
                corpora_vocab(
                    self._train_corpus,
                    self._test_corpus,
                    # case_insensitive=self._embedding.case_insensitive,
                )
            )
            include_tokens_path = join(self._gen_dir, "_incl_tokens.pkl")
            pickle_file(path=include_tokens_path, data=include)
            tokens_dict = tokenize_data(
                include=include,
                case_insensitive=self._embedding.case_insensitive,
                **to_tokenize,
            )
            for mode, token_data in tokens_dict.items():
                token_data_attr = "_{mode}_tokens".format(mode=mode)
                token_data_file = "_{mode}_tokens.pkl".format(mode=mode)
                token_data_path = join(self._gen_dir, token_data_file)
                pickle_file(path=token_data_path, data=token_data)
                setattr(self, token_data_attr, token_data)

    def _init_tfrecords(self):
        tokens_lists = {}
        for mode in ["train", "test"]:
            tfrecord_folder = "_{mode}".format(mode=mode)
            tfrecord_path = join(self._gen_dir, tfrecord_folder)
            if not exists(tfrecord_path):
                tokens_attr = "_{mode}_tokens".format(mode=mode)
                tokens_dict = getattr(self, tokens_attr)
                tokens_list = tokens_dict.values()
                tokens_lists[mode] = sum(tokens_list, [])
        fetch_results_path = join(self._gen_dir, "_fetch_results.pkl")
        if tokens_lists and not exists(fetch_results_path):
            vocab_file_templ = "_vocab{ext}"
            filtered_vocab_file = vocab_file_templ.format(ext=".filt.txt")
            filtered_vocab_path = join(self._gen_dir, filtered_vocab_file)
            if not exists(filtered_vocab_path):
                filtered_vocab = list(
                    set(self._vocab)
                    & set(
                        corpora_vocab(
                            self._train_corpus,
                            self._test_corpus,
                            # case_insensitive=self._embedding.case_insensitive,
                        )
                    )
                )
                indices = [self._vocab.index(word) for word in filtered_vocab]
                write_vocab_file(filtered_vocab_path, filtered_vocab, indices)
            #! There has to be at least 1 bucket for any
            #! test-time oov tokens (possibly targets)
            lookup_table = ids_lookup_table(
                filtered_vocab_path,
                self._num_oov_buckets,
                vocab_size=len(self._vocab),
            )
            fetch_dict = fetch_lookup_ops(lookup_table, **tokens_lists)
            fetch_results = run_lookups(fetch_dict, metadata_path=self.gen_dir)
            pickle_file(path=fetch_results_path, data=fetch_results)
        else:
            fetch_results = unpickle_file(fetch_results_path)
        oov_buckets = {}
        for mode, values in fetch_results.items():
            data_dict_attr = "_{mode}_dict".format(mode=mode)
            data_dict = getattr(self, data_dict_attr)
            string_features, int_features = split_list(values, parts=2)
            tfexamples = make_tf_examples(
                string_features, int_features, data_dict["labels"]
            )
            tfrecord_folder = "_{mode}".format(mode=mode)
            tfrecord_path = join(self._gen_dir, tfrecord_folder)
            if not exists(tfrecord_path):
                write_tfrecords(tfrecord_path, tfexamples)
            #! There has to be at least 1 bucket for any
            #! test-time oov tokens (possibly targets)
            buckets = [
                BUCKET_TOKEN.format(num=n + 1)
                for n in range(self._num_oov_buckets)
            ]
            oov_buckets[mode] = tokens_by_assigned_id(
                string_features,
                int_features,
                start=len(self._vocab),
                keys=buckets,
            )
        accum_oov_buckets = accumulate_dicts(
            **oov_buckets,
            accum_fn=lambda prev, curr: list(set(prev) | set(curr)),
        )
        self._oov_buckets = {
            buckets[i]: accum_oov_buckets[buckets[i]]
            for i in sorted(
                [buckets.index(key) for key in [*accum_oov_buckets]]
            )
        }
        # test_oov_file_path = join(self._gen_dir, "_test_oov.txt")
        # if not exists(test_oov_file_path):
        #     self._test_oov_vocab = sum(
        #         [
        #             [
        #                 word
        #                 for word in bucket
        #                 if word not in self._train_oov_vocab
        #             ]
        #             for bucket in self._oov_buckets.values()
        #         ],
        #         [],
        #     )
        #     write_vocab_file(test_oov_file_path, self._test_oov_vocab)
        # else:
        #     self._test_oov_vocab = read_vocab_file(test_oov_file_path)

    def _init_embedding_params(self):
        dim_size = self._embedding.dim_size
        vectors = self._embedding.vectors
        num_oov_vectors = len(self._vocab) - self._embedding.vocab_size
        num_oov_vectors += self._num_oov_buckets
        oov_fn = self._oov_fn or DEFAULT_OOV_FN
        if TF_RANDOM_SEED is not None:
            np.random.seed(TF_RANDOM_SEED)
        oov_vectors = oov_fn(size=(num_oov_vectors, dim_size))
        vectors = np.concatenate([vectors, oov_vectors], axis=0)
        vocab_size = len(vectors)
        num_shards = partitioner_num_shards(vocab_size)
        init_fn = embedding_initializer_fn(vectors, num_shards)
        self._embedding_params = {
            "_vocab_size": vocab_size,
            "_num_oov_buckets": self._num_oov_buckets,
            "_vocab_file": self._vocab_file,
            "_embedding_dim": dim_size,
            "_embedding_init": init_fn,
            "_embedding_num_shards": num_shards,
        }

    def _write_info_file(self):
        # TODO: check for existence and include a timestamp of last updated
        info_path = join(self.gen_dir, "info.json")
        info = {
            "uid": self.uid,
            "name": self.name,
            "datasets": {
                dataset.uid: {
                    "name": dataset.name,
                    "train": dataset.train_dist,
                    "test": dataset.test_dist,
                }
                for dataset in self.datasets
            },
            "oov_policy": {
                "oov": stringify(self._oov_fn),
                "oov_train_threshold": self._oov_train_threshold,
                "oov_buckets": self._oov_buckets,
            },
            "vocab_coverage": {**self._vocab_coverage()},
            "embedding": {
                "uid": self.embedding.uid,
                "name": self.embedding.name,
                "case_insensitive": self.embedding.case_insensitive,
                "filter_details": self.embedding.filter_info,
                "internal_params": {
                    k: stringify(v) for k, v in self._embedding_params.items()
                },
            },
        }
        dump_json(path=info_path, data=info)

    def _vocab_coverage(self):
        _ci = self._embedding.case_insensitive
        v_orig = set(self._embedding.vocab)
        v_extd = set(self._vocab)
        v_train = set(
            sum(
                accumulate_dicts(
                    self._train_tokens,
                    accum_fn=(lambda prev, curr: list(set(prev) | set(curr))),
                    default=lambda v=None: set(sum(v, [])) if v else set(),
                ).values(),
                [],
            )
        )
        v_test = set(
            sum(
                accumulate_dicts(
                    self._test_tokens,
                    accum_fn=(lambda prev, curr: list(set(prev) | set(curr))),
                    default=lambda v=None: set(sum(v, [])) if v else set(),
                ).values(),
                [],
            )
        )
        v_train_oov_over_t = (
            set(
                corpora_vocab(
                    self._test_corpus, threshold=self._oov_train_threshold
                )
            )
            - v_orig
        )
        v_tot = v_train | v_test
        v_oov = v_tot - v_orig

        n_tot = len(v_tot)
        n_oov = len(v_oov)
        n_train = len(v_train)
        n_test = len(v_test)
        n_train_oov = len(v_train - v_orig)
        n_train_oov_embd = len(v_train_oov_over_t)
        n_train_oov_bktd = len(v_train - v_extd)
        n_test_oov = len(v_test - v_orig)
        n_test_oov_bktd = len(v_test - v_extd)
        n_test_oov_excl = len(v_test - (v_extd | v_train))
        portion = lambda p, tot=None: str(p) + (
            " ({:.2f}%)".format((p / tot) * 100) if tot else ""
        )
        return {
            "total": {
                "size": n_tot,
                "in_vocab": portion(n_tot - n_oov, tot=n_tot),
                "out_of_vocab": portion(n_oov, tot=n_tot),
            },
            "train": {
                "size": n_train,
                "oov": {
                    "total": portion(n_train_oov, tot=n_train),
                    "embedded": portion(n_train_oov_embd, tot=n_train),
                    **(
                        {"bucketed": portion(n_train_oov_bktd, tot=n_train)}
                        if n_train_oov_bktd > 0
                        else {}
                    ),
                },
            },
            "test": {
                "size": n_test,
                "oov": {
                    "total": portion(n_test_oov, tot=n_test),
                    "bucketed": portion(n_test_oov_bktd, tot=n_test),
                    **(
                        {"exclusive": portion(n_test_oov_excl, tot=n_test)}
                        if n_train_oov_bktd > 0
                        else {}
                    ),
                },
            },
        }

    def _resolve_oov_fn(self, oov_arg):
        if not oov_arg:
            return DEFAULT_OOV_FN
        regexp = re.compile(r"^(?P<fn>[^\[]+)(\[(?P<args>[^\]]+)\])?")
        oov = regexp.search(oov_arg).groupdict()
        if oov.get("fn"):
            try:
                oov_fn = np.random.__dict__[oov["fn"]]
            except AttributeError:
                warn(
                    "Invalid oov function {}, using default".format(
                        oov.get("fn")
                    )
                )
                return DEFAULT_OOV_FN
        else:
            warn("No oov function provided, using default")
            return DEFAULT_OOV_FN
        if oov.get("args"):
            oov_args = [float(arg) for arg in oov["args"].split(",") if arg]
            oov_fn = partial(oov_fn, *oov_args)
            try:
                oov_fn(size=1)
                return oov_fn
            except TypeError:
                raise ValueError("Invalid OOV function arguments.")
