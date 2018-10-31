from os.path import join, exists
from os import makedirs
from csv import DictWriter
from datetime import datetime
from zipfile import ZipFile, ZIP_DEFLATED
from hashlib import md5
import math
import numpy as np
import spacy
from tqdm import tqdm
import tensorflow as tf
from tensorflow.train import BytesList, Feature, Features, Example, Int64List
from tensorflow.python.client.timeline import Timeline  # pylint: disable=E0611
from tensorflow.python_io import TFRecordWriter

from tsaplay.datasets import Dataset
from tsaplay.utils.io import pickle_file, unpickle_file
from tsaplay.utils.decorators import timeit
from tsaplay.utils.data import (
    class_dist_info,
    merge_dicts,
    class_dist_stats,
)
from tsaplay.constants import FEATURES_DATA_PATH, SPACY_MODEL, RANDOM_SEED


class FeatureProvider:
    def __init__(
        self, datasets, embedding, oov=None, max_shards=10, data_root=None
    ):
        np.random.seed(RANDOM_SEED)
        self._data_root = data_root or FEATURES_DATA_PATH
        self._embedding = embedding
        self._datasets = list(datasets)
        embedding_name = self._embedding.name
        datasets_name = [ds.name for ds in self._datasets]
        self._name = "--".join([embedding_name] + datasets_name)
        self._gen_dir = join(self._data_root, embedding_name, datasets_name)
        train_dict_path = join(self._gen_dir, "_train_dict.pkl")
        test_dict_path = join(self._gen_dir, "_test_dict.pkl")
        train_corpus_path = join(self.gen_dir, "_train_corpus.pkl")
        test_corpus_path = join(self.gen_dir, "_test_corpus.pkl")
        vocab_file_path = join(self.gen_dir, "_vocab_file.txt")
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
            pickle_file(path=test_corpus_path, data=self._test_corpus)
            test_docs = set(self._test_dict["sentences"])
            self._test_corpus = Dataset.generate_corpus(test_docs)
            pickle_file(path=test_corpus_path, data=self._test_corpus)
            self._dist_stats = {
                ds.name: class_dist_stats(
                    ds.class_labels, train=ds.train_dict, test=ds.test_dict
                )
                for ds in self._datasets
            }

            self.__fetch_dict = self._build_fetch_dict()
            if self.__fetch_dict:
                self._generate_missing_tf_record_files()

        self._oov = oov if callable(oov) else self.default_oov
        self._max_shards = max_shards

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
        return join(self._tf_record_folder("train"), "*.tfrecord")

    @property
    def test_tfrecords(self):
        return join(self._tf_record_folder("test"), "*.tfrecord")

    @classmethod
    def default_oov(cls, size):
        return np.random.uniform(low=-0.03, high=0.03, size=size)

    @property
    def embedding_params(self):
        return {
            "_vocab_size": self._embedding.vocab_size,
            "_vocab_file": self._embedding.vocab_file_path,
            "_embedding_dim": self._embedding.dim_size,
            "_embedding_init": self._embedding.initializer_fn,
            "_embedding_num_shards": self._embedding.num_shards,
        }

    @classmethod
    def ideal_partition_divisor(cls, vocab_size, max_shards):
        for i in range(max_shards, 0, -1):
            if vocab_size % i == 0:
                return i

    def embedding_initializer(self, structure=None):
        shape = (self.vocab_size, self.dim_size)
        partition_size = int(self.vocab_size / self.num_shards)

        def _init_var(shape=shape, dtype=tf.float32, partition_info=None):
            return self.vectors

        def _init_part_var(shape=shape, dtype=tf.float32, partition_info=None):
            part_offset = partition_info.single_offset(shape)
            this_slice = part_offset + partition_size
            return self.vectors[part_offset:this_slice]

        def _init_const():
            return self.vectors

        _init_fn = {
            "partitioned": _init_part_var,
            "constant": _init_const,
            "variable": _init_var,
        }.get(structure, _init_var)

        return _init_fn

    @classmethod
    @timeit("Tokenizing dataset", "Tokenization complete")
    def tokens_from_dict(cls, dictionary):
        offsets = dictionary.get("offsets", [])
        if not offsets:
            offsets = cls.get_target_offset_array(dictionary)
        dictionary = {**dictionary, "offsets": offsets}
        l_ctxts, trgs, r_ctxts = cls.partition_sentences(
            sentences=dictionary["sentences"],
            targets=dictionary["targets"],
            offsets=dictionary["offsets"],
        )
        l_tok = cls.tokenize_phrases(l_ctxts)
        trg_tok = cls.tokenize_phrases(trgs)
        r_tok = cls.tokenize_phrases(r_ctxts)

        return (l_tok, trg_tok, r_tok)

    @classmethod
    def get_target_offset_array(cls, dictionary):
        offsets = []
        dict_zip = zip(dictionary["sentences"], dictionary["targets"])
        for (sentence, target) in dict_zip:
            offsets.append(sentence.lower().find(target.lower()))
        return offsets

    @classmethod
    def tf_encode_tokens(cls, tokens):
        return "<SEP>".join(tokens).encode()

    @classmethod
    def get_tokens_sp_tensor(cls, tokens, delimiter="<SEP>"):
        tf_encoded_tokens = cls.tf_encode_tokens(tokens)
        string_tensor = tf.constant([tf_encoded_tokens], dtype=tf.string)
        tokens_sp_tensor = tf.string_split(string_tensor, delimiter)
        return tokens_sp_tensor

    @classmethod
    def tf_lookup_string_sp(cls, table, tokens_sp_tensor):
        ids_sp_tensor = table.lookup(tokens_sp_tensor)
        ids_tensor = tf.sparse_tensor_to_dense(ids_sp_tensor)
        return ids_tensor

    @classmethod
    def partition_sentences(cls, sentences, targets, offsets):
        left_ctxts, _targets, right_ctxts = [], [], []

        for (sen, trg, off) in zip(sentences, targets, offsets):
            left_ctxts.append(sen[:off].strip())
            target = trg.strip()
            r_off = off + len(target)
            _targets.append(target)
            right_ctxts.append(sen[r_off:].strip())
        return left_ctxts, _targets, right_ctxts

    @classmethod
    def zero_norm_labels(cls, labels):
        minimum = abs(min(labels))
        return [l + minimum for l in labels]

    @classmethod
    def index_lookup_table(cls, vocab_file, num_oov=0):
        return tf.contrib.lookup.index_table_from_file(
            vocabulary_file=vocab_file,
            key_column_index=0,
            value_column_index=1,
            num_oov_buckets=num_oov,
            delimiter="\t",
        )

    @classmethod
    def token_filter(cls, token):
        if token.like_url:
            return False
        if token.like_email:
            return False
        if token.text in ["\uFE0F"]:
            return False
        return True

    @classmethod
    def tokenize_phrases(cls, phrases):
        token_lists = []
        nlp = spacy.load(SPACY_MODEL, disable=["parser", "ner"])
        for doc in tqdm(
            nlp.pipe(phrases, batch_size=100, n_threads=-1), total=len(phrases)
        ):
            tokens = list(filter(cls.token_filter, doc))
            token_lists.append([t.text.lower() for t in tokens])
        return token_lists

    # @classmethod
    # @timeit("Generating vocabulary for lookup table", "Vocabulary generated")
    # def generate_lookup_vocab(cls, embedding, datasets):
    #     datasets_corpus = [
    #         [word.lower() for word in dataset.corpus]
    #         for dataset in list(datasets)
    #     ]
    #     datasets_corpus = sum(datasets_corpus, [])
    #     not_oov_words = set(embedding.vocab) & set(datasets_corpus)
    #     num_oov_words = len(datasets_corpus) - len(not_oov_words)
    #     vocab_lookup_data = [
    #         (word, embedding.vocab.index(word)) for word in not_oov_words
    #     ]

    #     return vocab_lookup_data, num_oov_words

    def _tf_record_folder(self, mode):
        return join(self.gen_dir, "_{mode}".format(mode=mode))

    def _write_tf_record_files(self, mode, tf_examples):
        np.random.shuffle(tf_examples)
        num_per_shard = int(math.ceil(len(tf_examples) / float(10)))
        total_shards = int(math.ceil(len(tf_examples) / float(num_per_shard)))
        folder_path = self._tf_record_folder(mode)
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

        return folder_path

    def _build_fetch_dict(self):
        vocab_file_data, num_oov_words = self.generate_lookup_vocab(
            self._embedding, self._datasets
        )
        vocab_file_path = join(self.gen_dir, "_vocab_file.txt")
        with open(vocab_file_path, "w") as vocab_file:
            for (word, index) in vocab_file_data:
                vocab_file.write("{0}\t{1}\n".format(word, index))
        ids_table = self.index_lookup_table(vocab_file_path, num_oov_words)
        return {
            mode: self._append_fetches("train", ids_table)
            for mode in ["trian", "test"]
        }

    def _feature_lists_from_dict(self, feats):
        feats["left"] = [l.tolist()[0] for l in feats["left"]]
        feats["target"] = [t.tolist()[0] for t in feats["target"]]
        feats["right"] = [r.tolist()[0] for r in feats["right"]]
        feats["left_ids"] = [l.tolist()[0] for l in feats["left_ids"]]
        feats["target_ids"] = [t.tolist()[0] for t in feats["target_ids"]]
        feats["right_ids"] = [r.tolist()[0] for r in feats["right_ids"]]
        feature_lists = (feats[k] for k in [*feats])
        return feature_lists

    def _make_tf_example(self, l, trg, r, l_ids, t_ids, r_ids, label):
        feature = {
            "left": Feature(bytes_list=BytesList(value=l)),
            "target": Feature(bytes_list=BytesList(value=trg)),
            "right": Feature(bytes_list=BytesList(value=r)),
            "left_ids": Feature(int64_list=Int64List(value=l_ids)),
            "target_ids": Feature(int64_list=Int64List(value=t_ids)),
            "right_ids": Feature(int64_list=Int64List(value=r_ids)),
            "labels": Feature(int64_list=Int64List(value=[label])),
        }
        tf_example = Example(features=Features(feature=feature))
        return tf_example.SerializeToString()

    @timeit("Generating any missing tfrecord files", "TFrecord files ready")
    def _generate_missing_tf_record_files(self):
        values = self._run_fetches()
        for mode in ["train", "test"]:
            if mode == "train":
                labels = self._train_dict["labels"]
            else:
                labels = self._test_dict["labels"]
            labels = self.zero_norm_labels(labels)
            features = self._feature_lists_from_dict(values[mode])
            data_zip = zip(*features, labels)
            tf_examples = [self._make_tf_example(*dz) for dz in data_zip]
            self._write_tf_record_files(mode, tf_examples)

    @timeit("Generating sparse tensors of tokens", "Sparse tensors generated")
    def _sparse_tensors_from_tokens(self, l_tok, trg_tok, r_tok):
        l_sp = [self.get_tokens_sp_tensor(l) for l in tqdm(l_tok)]
        trg_sp = [self.get_tokens_sp_tensor(t) for t in tqdm(trg_tok)]
        r_sp = [self.get_tokens_sp_tensor(r) for r in tqdm(r_tok)]
        return (l_sp, trg_sp, r_sp)

    @timeit("Building graph with required embedding lookup ops", "Graph built")
    def _append_fetches(self, mode, ids_table):
        if mode == "train":
            data = self._train_dict
        else:
            data = self._test_dict

        tokens = self.tokens_from_dict(data)
        self._write_tokens_file(mode, tokens)

        sparse_tokens = self._sparse_tensors_from_tokens(*tokens)

        left_ids_ops, target_ids_ops, right_ids_ops = [], [], []
        left_ops, target_ops, right_ops = [], [], []

        for (l, t, r) in zip(*sparse_tokens):
            left_ids_ops.append(self.tf_lookup_string_sp(ids_table, l))
            target_ids_ops.append(self.tf_lookup_string_sp(ids_table, t))
            right_ids_ops.append(self.tf_lookup_string_sp(ids_table, r))

            left_ops.append(tf.sparse_tensor_to_dense(l, default_value=b""))
            target_ops.append(tf.sparse_tensor_to_dense(t, default_value=b""))
            right_ops.append(tf.sparse_tensor_to_dense(r, default_value=b""))

        return {
            "left": left_ops,
            "target": target_ops,
            "right": right_ops,
            "left_ids": left_ids_ops,
            "target_ids": target_ids_ops,
            "right_ids": right_ids_ops,
        }

    @timeit("Executing graph", "Graph execution complete")
    def _run_fetches(self, write_metadata=False):
        if tf.executing_eagerly():
            raise ValueError("Eager execution is not supported.")
        with tf.Session() as sess:
            sess.run(tf.tables_initializer())
            run_opts = (
                tf.RunOptions(
                    trace_level=tf.RunOptions.FULL_TRACE  # pylint: disable=E1101
                )
                if write_metadata
                else None
            )
            run_metadata = tf.RunMetadata()
            values = sess.run(
                self.__fetch_dict, options=run_opts, run_metadata=run_metadata
            )
        if write_metadata:
            self._write_run_metadata(run_metadata)
        return values

    @timeit("Exporting graph run metadata", "Metadata exported")
    def _write_run_metadata(self, run_metadata):
        file_dir = join(self._data_root, "_meta")
        makedirs(file_dir, exist_ok=True)
        file_name = datetime.now().strftime("%Y%m%d-%H%M%S") + ".json"
        time_line = Timeline(run_metadata.step_stats)  # pylint: disable=E1101
        ctf = time_line.generate_chrome_trace_format()
        zip_name = file_name.replace(".json", ".zip")
        zip_path = join(file_dir, zip_name)
        with ZipFile(zip_path, "w", ZIP_DEFLATED) as zipf:
            zipf.writestr(file_name, data=ctf)

    @timeit("Exporting tokens to csv file", "Tokens csv exported")
    def _write_tokens_file(self, mode, tokens):
        file_name = "_" + mode + "_tokens.csv"
        file_path = join(self.gen_dir, file_name)
        with open(file_path, "w", encoding="utf-8") as csvfile:
            fieldnames = ["Left", "Target", "Right"]
            writer = DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for (left, target, right) in zip(*tokens):
                writer.writerow(
                    {
                        "Left": " ".join(left),
                        "Target": " ".join(target),
                        "Right": " ".join(right),
                    }
                )
        return file_path
