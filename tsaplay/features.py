from os.path import join, exists, dirname
from os import makedirs, walk
import numpy as np

from tsaplay.constants import (
    FEATURES_DATA_PATH,
    DEFAULT_OOV_FN,
    BUCKET_TOKEN,
    RANDOM_SEED,
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
)


class FeatureProvider:
    def __init__(self, datasets, embedding, **kwargs):
        num_oov_buckets = kwargs.get("num_oov_buckets", 0)
        oov = kwargs.get("oov")
        self._num_oov_buckets = max(num_oov_buckets, 0)
        self._oov_fn = (
            DEFAULT_OOV_FN
            if ((oov or self._num_oov_buckets) and not callable(oov))
            else oov
        )

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
        self._train_tokens = None
        self._test_tokens = None
        self._train_tfrecords = None
        self._test_tfrecords = None
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

    def _init_uid(self):
        datasets_uids = [dataset.uid for dataset in self.datasets]
        dataset_names = [dataset.name for dataset in self.datasets]
        oov_policy = [bool(self._oov_fn), self._num_oov_buckets]
        uid_data = [self._embedding.uid] + datasets_uids + oov_policy
        print(uid_data)
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
                getattr(dataset, corpus_attr) for dataset in self.datasets
            )
            corpus = merge_corpora(*corpora)
            pickle_file(path=corpus_path, data=corpus)
        setattr(self, corpus_attr, corpus)

    def _init_vocab(self):
        vocab_file_templ = "_vocab{ext}"
        vocab_file = vocab_file_templ.format(ext=".txt")
        vocab_file_path = join(self._gen_dir, vocab_file)
        self._vocab_file = vocab_file_path
        if exists(self._vocab_file):
            self._vocab = read_vocab_file(vocab_file_path)
        else:
            self._vocab = self._embedding.vocab
            if self._oov_fn and not self._num_oov_buckets:
                train_vocab = set(
                    corpora_vocab(
                        self._train_corpus,
                        case_insensitive=self._embedding.case_insensitive,
                    )
                )
                train_oov_vocab = list(train_vocab - set(self._vocab))
                train_oov_vocab.sort()
                self._vocab += train_oov_vocab
            write_vocab_file(vocab_file_path, self._vocab)
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
            include = set(self._vocab) | (
                set(
                    corpora_vocab(
                        self._train_corpus,
                        self._test_corpus,
                        case_insensitive=self._embedding.case_insensitive,
                    )
                )
                if self._num_oov_buckets
                else set()
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
                tokens_list = [value for value in tokens_dict.values()]
                tokens_lists[mode] = sum(tokens_list, [])
        if tokens_lists:
            fetch_results_path = join(self._gen_dir, "_fetch_results.pkl")
            if not exists(fetch_results_path):
                vocab_file_templ = "_vocab{ext}"
                filtered_vocab_file = vocab_file_templ.format(ext=".filt.txt")
                filtered_vocab_path = join(self._gen_dir, filtered_vocab_file)
                if not exists(filtered_vocab_path):
                    filtered_vocab = set(self._vocab) & set(
                        corpora_vocab(
                            self._train_corpus,
                            self._test_corpus,
                            case_insensitive=self._embedding.case_insensitive,
                        )
                    )
                    indices = [
                        self._vocab.index(word) for word in filtered_vocab
                    ]
                    write_vocab_file(
                        filtered_vocab_path, filtered_vocab, indices
                    )
                lookup_table = ids_lookup_table(
                    filtered_vocab_path, self._num_oov_buckets
                )
                fetch_dict = fetch_lookup_ops(lookup_table, **tokens_lists)
                fetch_results = run_lookups(
                    fetch_dict, metadata_path=self.gen_dir
                )
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
                write_tfrecords(tfrecord_path, tfexamples)
                if self._num_oov_buckets:
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
            if oov_buckets:
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

    def _init_embedding_params(self):
        np.random.seed(RANDOM_SEED)
        dim_size = self._embedding.dim_size
        vectors = self._embedding.vectors
        num_oov_vectors = len(self._vocab) - self._embedding.vocab_size
        num_oov_vectors += self._num_oov_buckets
        if num_oov_vectors:
            oov_fn = self._oov_fn or DEFAULT_OOV_FN
            oov_vectors = oov_fn(size=(num_oov_vectors, dim_size))
            vectors = np.concatenate([vectors, oov_vectors], axis=0)
        vocab_size = len(vectors)
        num_shards = partitioner_num_shards(vocab_size)
        init_fn = embedding_initializer_fn(vectors, num_shards)
        self._embedding_params = {
            "_vocab_size": vocab_size,
            "_vocab_file": self._vocab_file,
            "_embedding_dim": dim_size,
            "_embedding_init": init_fn,
            "_embedding_num_shards": num_shards,
        }

    def _write_info_file(self):
        info_path = join(self.gen_dir, "info.json")
        info = {
            "uid": self.uid,
            "name": self.name,
            "datasets": {
                "uids": [dataset.uid for dataset in self.datasets],
                "names": [dataset.name for dataset in self.datasets],
            },
            "embedding": {
                "uid": self.embedding.uid,
                "name": self.embedding.name,
            },
            "oov_policy": {
                "oov": stringify(self._oov_fn),
                "num_oov_buckets": self._num_oov_buckets,
                "oov_buckets": self._oov_buckets,
            },
        }
        dump_json(path=info_path, data=info)
