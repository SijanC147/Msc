from os.path import join, exists
from os import makedirs
from csv import DictWriter
from hashlib import md5
import math
import numpy as np
import spacy
import collections
from shutil import rmtree
from tqdm import tqdm
import tensorflow as tf
from tensorflow.train import BytesList, Feature, Features, Example, Int64List
from tensorflow.python_io import TFRecordWriter

from tsaplay.datasets import Dataset
from tsaplay.utils.filters import default_token_filter
from tsaplay.utils.io import (
    pickle_file,
    unpickle_file,
    export_run_metadata,
    write_csv,
)
from tsaplay.utils.decorators import timeit
from tsaplay.utils.data import (
    merge_dicts,
    class_dist_stats,
    partition_sentences,
    zero_norm_labels,
    split_list,
)
from tsaplay.constants import (
    FEATURES_DATA_PATH,
    SPACY_MODEL,
    RANDOM_SEED,
    DELIMITER,
    DEFAULT_OOV_FN,
)


class FeatureProvider:
    def __init__(
        self,
        datasets,
        embedding,
        oov=None,
        max_shards=10,
        oov_buckets=0,
        data_root=None,
    ):
        np.random.seed(RANDOM_SEED)
        self._data_root = data_root or FEATURES_DATA_PATH
        self._embedding = embedding
        self._datasets = (
            datasets
            if isinstance(datasets, collections.Iterable)
            else [datasets]
        )
        embedding_name = self._embedding.name
        datasets_name = "--".join([ds.name for ds in self._datasets])
        self._name = "{0}--{1}".format(embedding_name, datasets_name)
        self._gen_dir = join(self._data_root, embedding_name, datasets_name)
        self._tfrecord_path = join(self._gen_dir, "_{mode}")
        self._tfrecords_glob = join(self._tfrecord_path, "*.tfrecord")
        train_dict_path = join(self._gen_dir, "_train_dict.pkl")
        test_dict_path = join(self._gen_dir, "_test_dict.pkl")
        train_corpus_path = join(self._gen_dir, "_train_corpus.pkl")
        test_corpus_path = join(self._gen_dir, "_test_corpus.pkl")
        vocab_file_path = join(self._gen_dir, "_vocab_file.txt")
        if exists(self._gen_dir):
            rmtree(self._gen_dir)
        if not exists(self._gen_dir):
            train_dicts = (ds.train_dict for ds in self._datasets)
            self._train_dict = merge_dicts(*train_dicts)
            pickle_file(path=train_dict_path, data=self._train_dict)
            test_dicts = (ds.test_dict for ds in self._datasets)
            self._test_dict = merge_dicts(*test_dicts)
            pickle_file(path=test_dict_path, data=self._test_dict)
            labels = self._train_dict["labels"] + self._test_dict["labels"]
            self._class_labels = list(set(labels))
            train_docs = set(self._train_dict["sentences"])
            self._train_corpus = Dataset.generate_corpus(train_docs)
            pickle_file(path=train_corpus_path, data=self._train_corpus)
            test_docs = set(self._test_dict["sentences"])
            self._test_corpus = Dataset.generate_corpus(test_docs)
            pickle_file(path=test_corpus_path, data=self._test_corpus)
            self._dist_stats = {
                ds.name: class_dist_stats(
                    ds.class_labels, train=ds.train_dict, test=ds.test_dict
                )
                for ds in self._datasets
            }
            train_vocab = set(word.lower() for word in [*self._train_corpus])
            embedding_vocab = set(self._embedding.vocab)
            in_vocab_words = embedding_vocab & train_vocab
            oov_words = train_vocab - embedding_vocab
            if oov:
                oov = (
                    DEFAULT_OOV_FN
                    if not callable(oov) or oov == "default"
                    else oov
                )
                self._embedding.vocab += list(oov_words)
                embedding_dim = self._embedding.dim_size
                embedding_vectors = self._embedding.vectors
                oov_vectors = np.asarray(
                    [oov(size=embedding_dim) for word in oov_words]
                ).astype(np.float32)
                self._embedding.vectors = np.concatenate(
                    [embedding_vectors, oov_vectors]
                )
                vocab_lookup_info = [
                    (word, self._embedding.vocab.index(word))
                    for word in train_vocab
                ]
            else:
                vocab_lookup_info = [
                    (word, self._embedding.vocab.index(word))
                    for word in in_vocab_words
                ]
            with open(vocab_file_path, "w") as vocab_file:
                for (word, index) in vocab_lookup_info:
                    vocab_file.write("{0}\t{1}\n".format(word, index))
            lookup_table = self.index_lookup_table(
                vocab_file_path, oov_buckets
            )
            tokenized_data = self.tokenize_data(
                train=self._train_dict,
                test=self._test_dict,
                exclude=oov_words if not oov else None,
                export_dir=self._gen_dir,
            )
            fetch_dict = self._make_fetch_dict(tokenized_data, lookup_table)
            values = self._fetch_data(fetch_dict)
            self._generate_tf_records(values)

        for i in range(max_shards, 0, -1):
            if self._embedding.vocab_size % i == 0:
                self._embedding_shards = i

    @property
    def name(self):
        return self._name

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
    def dist_stats(self):
        return self._dist_stats

    @property
    def train_tfrecords(self):
        return self._tfrecords_glob.format(mode="train")

    @property
    def test_tfrecords(self):
        return self._tfrecords_glob.format(mode="test")

    @property
    def embedding_params(self):
        return {
            "_vocab_size": self._embedding.vocab_size,
            "_vocab_file": self._embedding.vocab_file_path,
            "_embedding_dim": self._embedding.dim_size,
            "_embedding_init": self.embedding_initializer,
            "_embedding_num_shards": self._embedding_shards,
        }

    def embedding_initializer(self, structure=None):
        embedding = self._embedding
        shape = (embedding.vocab_size, embedding.dim_size)
        partition_size = int(embedding.vocab_size / self._embedding_shards)

        def _init_var(shape=shape, dtype=tf.float32, partition_info=None):
            return embedding.vectors

        def _init_part_var(shape=shape, dtype=tf.float32, partition_info=None):
            part_offset = partition_info.single_offset(shape)
            this_slice = part_offset + partition_size
            return embedding.vectors[part_offset:this_slice]

        def _init_const():
            return embedding.vectors

        _init_fn = {
            "partitioned": _init_part_var,
            "constant": _init_const,
            "variable": _init_var,
        }.get(structure, _init_var)

        return _init_fn

    @classmethod
    def index_lookup_table(cls, vocab_file, oov_buckets=0):
        return tf.contrib.lookup.index_table_from_file(
            vocabulary_file=vocab_file,
            key_column_index=0,
            value_column_index=1,
            num_oov_buckets=oov_buckets,
            delimiter="\t",
        )

    @classmethod
    def tokenize_data(cls, exclude=None, export_dir=None, **data_dicts):
        exclude = set([word.lower() for word in exclude] if exclude else [])
        token_jobs = [
            sum(
                partition_sentences(
                    data_dict["sentences"], data_dict["targets"]
                ),
                [],
            )
            for data_dict in data_dicts.values()
        ]
        job_counts = [len(token_job) for token_job in token_jobs]
        token_jobs_joined = sum(token_jobs, [])
        nlp = spacy.load(SPACY_MODEL, disable=["parser", "ner"])
        token_generator = nlp.pipe(
            token_jobs_joined, batch_size=100, n_threads=-1
        )
        encoded_tokens = [
            DELIMITER.join(
                [
                    token.text.lower()
                    for token in list(filter(default_token_filter, doc))
                    if token.text.lower() not in exclude
                ]
            ).encode()
            for doc in tqdm(token_generator, total=len(token_jobs_joined))
        ]
        tokens_dict = {
            key: tokens
            for key, tokens in zip(
                [*data_dicts], split_list(encoded_tokens, counts=job_counts)
            )
        }
        if export_dir:
            csv_file_name = "_{}_tokens.csv"
            for mode, tokens_list in tokens_dict.items():
                csv_file_path = join(export_dir, csv_file_name.format(mode))
                decoded_tokens = [
                    [
                        str(enc_token, "utf-8").replace(DELIMITER, " ")
                        for enc_token in enc_tokens
                    ]
                    for enc_tokens in split_list(tokens_list, parts=3)
                ]
                csv_data_dict = {
                    key: decoded_token
                    for key, decoded_token in zip(
                        ["Left", "Target", "Right"], decoded_tokens
                    )
                }
                write_csv(csv_file_path, csv_data_dict)

        return tokens_dict

    def _make_fetch_dict(self, tokenized_data_dict, lookup_table):
        fetch_dict = {}
        for (key, value) in tokenized_data_dict.items():
            sp_tensors = [
                tf.string_split(
                    tf.constant([tokens_list], dtype=tf.string), DELIMITER
                )
                for tokens_list in tqdm(value)
            ]
            string_ops = [
                tf.sparse_tensor_to_dense(sp_tensor, default_value=b"")
                for sp_tensor in sp_tensors
            ]
            id_ops = [
                tf.sparse_tensor_to_dense(lookup_table.lookup(sp_tensor))
                for sp_tensor in sp_tensors
            ]
            fetch_dict[key] = string_ops + id_ops
        return fetch_dict

    @timeit("Executing graph", "Graph execution complete")
    def _fetch_data(self, fetch_dict, write_metadata=False):
        run_opts = (
            tf.RunOptions(
                trace_level=tf.RunOptions.FULL_TRACE  # pylint: disable=E1101
            )
            if write_metadata
            else None
        )
        run_metadata = tf.RunMetadata()
        if tf.executing_eagerly():
            raise ValueError("Eager execution is not supported.")
        with tf.Session() as sess:
            sess.run(tf.tables_initializer())
            values_dict = sess.run(
                fetch_dict, options=run_opts, run_metadata=run_metadata
            )
        if write_metadata:
            metadata_path = join(self._data_root, "_meta")
            makedirs(metadata_path, exist_ok=True)
            export_run_metadata(run_metadata, path=metadata_path)
        values_lists = {}
        for (key, values) in values_dict.items():
            values_lists[key] = [value.tolist()[0] for value in values]
        return values_lists

    def _generate_tf_records(self, values_dict):
        for (mode, values) in values_dict.items():
            data_dict = "_{mode}_dict".format(mode=mode)
            labels = self.__getattribute__(data_dict)["labels"]
            string_features, int_features = split_list(values, parts=2)
            int_features += [[label] for label in zero_norm_labels(labels)]
            string_features = [
                Feature(bytes_list=BytesList(value=val))
                for val in string_features
            ]
            int_features = [
                Feature(int64_list=Int64List(value=val))
                for val in int_features
            ]
            all_features = string_features + int_features
            features_list = [
                Features(
                    feature={
                        "left": left,
                        "target": target,
                        "right": right,
                        "left_ids": left_ids,
                        "target_ids": target_ids,
                        "right_ids": right_ids,
                        "labels": label,
                    }
                )
                for (
                    left,
                    target,
                    right,
                    left_ids,
                    target_ids,
                    right_ids,
                    label,
                ) in zip(*split_list(all_features, parts=7))
            ]
            tf_examples = [
                Example(features=features).SerializeToString()
                for features in features_list
            ]
            self._write_tf_record_files(mode, tf_examples)

    def _write_tf_record_files(self, mode, tf_examples):
        np.random.shuffle(tf_examples)
        num_per_shard = int(math.ceil(len(tf_examples) / float(10)))
        total_shards = int(math.ceil(len(tf_examples) / float(num_per_shard)))
        folder_path = self._tfrecord_path.format(mode=mode)
        makedirs(folder_path, exist_ok=True)
        for shard_no in range(total_shards):
            start = shard_no * num_per_shard
            end = min((shard_no + 1) * num_per_shard, len(tf_examples))
            file_name = "{0}_of_{1}.tfrecord".format(
                shard_no + 1, total_shards
            )
            file_path = join(folder_path, file_name)
            with TFRecordWriter(file_path) as tf_writer:
                for serialized_example in tf_examples[start:end]:
                    tf_writer.write(serialized_example)
            if end == len(tf_examples):
                break
