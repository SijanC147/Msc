from os.path import join, exists
from os import makedirs
import numpy as np

from tsaplay.constants import FEATURES_DATA_PATH, DEFAULT_OOV_FN
from tsaplay.utils.tf import (
    index_lookup_table,
    fetch_lookup_ops,
    run_lookups,
    make_tf_examples,
    partitioner_num_shards,
    embedding_initializer,
)
from tsaplay.utils.io import (
    pickle_file,
    unpickle_file,
    write_tfrecords,
    write_vocab_file,
    read_vocab_file,
)
from tsaplay.utils.data import (
    merge_dicts,
    merge_corpi,
    split_list,
    hash_data,
    tokenize_data,
)


class FeatureProvider:
    def __init__(self, datasets, embedding, oov=None, oov_buckets=0):
        self._name = None
        self._uid = None
        self._gen_dir = None
        self._vocab = None
        self._datasets = None
        self._embedding = None
        self._class_labels = None
        self._train_dict = None
        self._test_dict = None
        self._train_corpus = None
        self._test_corpus = None
        self._train_tokens = None
        self._test_tokens = None
        self._train_tfrecords = None
        self._test_tfrecords = None

        self._init_uid(datasets, embedding, oov, oov_buckets)
        self._init_gen_dir(self.uid)
        for mode in ["train", "test"]:
            self._init_data_dict(mode, datasets)
            self._init_corpus(mode, datasets)

        temp_vocab_file_path = self._init_vocab(embedding, oov)

        self._init_token_data()
        self._init_tfrecords(temp_vocab_file_path, oov_buckets)
        self._init_embedding_params(embedding, oov)

    @property
    def name(self):
        return self._name

    @property
    def uid(self):
        return self._uid

    @property
    def datasets(self):
        return self._datasets

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

    def _init_uid(self, datasets, embedding, oov, oov_buckets):
        try:
            datasets_uids = [dataset.uid for dataset in datasets]
        except TypeError:
            datasets_uids = [dataset.uid for dataset in [datasets]]
        oov_policy = [oov, oov_buckets]
        uid_data = [embedding.uid] + datasets_uids + oov_policy
        self._uid = hash_data(uid_data)

    def _init_gen_dir(self, uid):
        data_root = FEATURES_DATA_PATH
        gen_dir = join(data_root, uid)
        if not exists(gen_dir):
            makedirs(gen_dir)
        self._gen_dir = gen_dir

    def _init_data_dict(self, mode, datasets):
        data_dict_attr = "_{mode}_dict".format(mode=mode)
        data_dict_file = "_{mode}_dict.pkl".format(mode=mode)
        data_dict_path = join(self._gen_dir, data_dict_file)
        if exists(data_dict_path):
            data_dict = unpickle_file(data_dict_path)
        else:
            try:
                data_dicts = (
                    getattr(dataset, data_dict_attr) for dataset in datasets
                )
                data_dict = merge_dicts(*data_dicts)
            except TypeError:
                data_dict = getattr(datasets, data_dict_attr)
        class_labels = self._class_labels or []
        class_labels = set(class_labels + data_dict["labels"])
        self._class_labels = list(class_labels)
        setattr(self, data_dict_attr, data_dict)

    def _init_corpus(self, mode, datasets):
        corpus_attr = "_{mode}_corpus".format(mode=mode)
        corpus_file = "_{mode}_corpus.pkl".format(mode=mode)
        corpus_path = join(self._gen_dir, corpus_file)
        if exists(corpus_path):
            corpus = unpickle_file(corpus_path)
        else:
            try:
                corpi = (getattr(dataset, corpus_attr) for dataset in datasets)
                corpus = merge_corpi(*corpi)
            except TypeError:
                corpus = getattr(datasets, corpus_attr)
        setattr(self, corpus_attr, corpus)

    def _init_vocab(self, embedding, oov):
        vocab_file = "_vocab{ext}"
        vocab_path = join(self._gen_dir, vocab_file.format(ext=".txt"))
        self._vocab_file = vocab_path
        if exists(self._vocab_file):
            self._vocab = read_vocab_file(vocab_path)
        else:
            self._vocab = embedding.vocab
            train_vocab = map(str.lower, [*self._train_corpus])
            if oov:
                oov_vocab = set(train_vocab) - set(self._vocab)
                self._vocab += list(oov_vocab)
            write_vocab_file(vocab_path, self._vocab)
            vocab_tsv = join(self._gen_dir, vocab_file.format(ext=".tsv"))
            write_vocab_file(vocab_tsv, self._vocab)
        temp_vocab_file = vocab_path.format(ext=".filt.txt")
        if not exists(temp_vocab_file):
            temp_vocab = list(set(train_vocab) + set(self._vocab))
            indices = [self._vocab.index(word) for word in temp_vocab]
            write_vocab_file(temp_vocab_file, temp_vocab, indices=indices)
        return temp_vocab_file

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
            tokens_dict = tokenize_data(include=self._vocab, **to_tokenize)
            for mode, token_data in tokens_dict.items():
                token_data_attr = "_{mode}_tokens".format(mode=mode)
                token_data_file = "_{mode}_tokens.pkl".format(mode=mode)
                token_data_path = join(self._gen_dir, token_data_file)
                pickle_file(path=token_data_path, data=token_data)
                setattr(self, token_data_attr, token_data)

    def _init_tfrecords(self, vocab_file, oov_buckets):
        fetches = {}
        lookup_table = index_lookup_table(vocab_file, oov_buckets)
        for mode in ["train", "test"]:
            tfrecord_folder = "_{mode}".format(mode=mode)
            tfrecord_path = join(self._gen_dir, tfrecord_folder)
            if not exists(tfrecord_path):
                tokens_attr = "_{mode}_tokens"
                tokens_dict = getattr(self, tokens_attr)
                tokens_lists = [value for value in tokens_dict.values()]
                tokens_lists = sum(tokens_lists, [])
                fetches[mode] = fetch_lookup_ops(tokens_lists, lookup_table)
        if fetches:
            fetch_results = run_lookups(vocab_file, metadata_path=self.gen_dir)
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

    def _init_embedding_params(self, embedding, oov):
        vocab_size = len(self._vocab)
        num_shards = partitioner_num_shards(vocab_size)
        dim_size = embedding.dim_size
        num_oov_words = vocab_size - embedding.vocab_size
        oov = oov or DEFAULT_OOV_FN
        embedding_vectors = embedding.vectors
        oov_vectors = oov(size=(num_oov_words, dim_size))
        vectors = np.concatenate([embedding_vectors, oov_vectors], axis=0)
        init_fn = embedding_initializer(vectors, num_shards)
        self._embedding_params = {
            "_vocab_size": vocab_size,
            "_vocab_file": self._vocab_file,
            "_embedding_dim": dim_size,
            "_embedding_init": init_fn,
            "_embedding_num_shards": num_shards,
        }
