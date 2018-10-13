from os.path import join, exists
from os import makedirs
from csv import DictWriter
from datetime import datetime
from zipfile import ZipFile, ZIP_DEFLATED
import math
import numpy as np
import spacy
from tqdm import tqdm
import tensorflow as tf
from tensorflow.train import BytesList, Feature, Features, Example, Int64List
from tensorflow.python.client.timeline import Timeline  # pylint: disable=E0611
from tensorflow.python_io import TFRecordWriter

from tsaplay.utils.decorators import timeit
from tsaplay.utils.data import get_class_distribution
from tsaplay.constants import FEATURES_DATA_PATH, SPACY_MODEL


class FeatureProvider:
    def __init__(self, datasets, embedding, num_shards=None, data_root=None):
        self._data_root = data_root or FEATURES_DATA_PATH
        self._embedding = embedding
        self._datasets = datasets
        self._num_shards = num_shards or 10
        self.__fetch_dict = self._build_fetch_dict()
        if self.__fetch_dict:
            self._generate_missing_tf_record_files()

    @property
    def name(self):
        dataset_names = [
            dataset.name + dataset.get_dist_key() for dataset in self._datasets
        ]
        return "--".join([self._embedding.name] + dataset_names)

    @property
    def embedding_params(self):
        return {
            "_vocab_size": self._embedding.vocab_size,
            "_vocab_file": self._embedding.vocab_file_path,
            "_embedding_dim": self._embedding.dim_size,
            "_embedding_init": self._embedding.initializer,
        }

    @property
    def train_tfrecords(self):
        return [
            self._get_tf_record_folder_pattern(dataset, "train")
            for dataset in self._datasets
        ]

    @property
    def test_tfrecords(self):
        return [
            self._get_tf_record_folder_pattern(dataset, "test")
            for dataset in self._datasets
        ]

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
        for (s, t) in zip(dictionary["sentences"], dictionary["targets"]):
            offsets.append(s.lower().find(t.lower()))
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
    def index_lookup_table(cls, vocab_file):
        return tf.contrib.lookup.index_table_from_file(
            vocabulary_file=vocab_file,
            key_column_index=0,
            value_column_index=1,
            default_value=1,
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
        for doc in tqdm(nlp.pipe(phrases, batch_size=100, n_threads=-1)):
            tokens = list(filter(cls.token_filter, doc))
            token_lists.append([t.text.lower() for t in tokens])
        return token_lists

    def get_datasets_stats(self):
        stats = {}
        for dataset in self._datasets:
            stats[dataset.name] = stats.get(dataset.name, {})
            stats[dataset.name].update(
                dataset.get_stats_dict(
                    dataset.default_classes,
                    train=dataset.train_dict,
                    test=dataset.test_dict,
                )
            )
        return stats

    def get_unique_classes(self):
        train_classes = np.array(
            [get_class_distribution(ds.train_dict)[0] for ds in self._datasets]
        ).flatten()
        test_classes = np.array(
            [get_class_distribution(ds.test_dict)[0] for ds in self._datasets]
        ).flatten()

        classes = np.unique(
            np.concatenate([train_classes, test_classes], axis=0)
        )
        classes = classes.astype(np.str).tolist()

        return classes

    def _get_gen_dir(self, dataset, mode=None):
        dataset_dir = join(self._data_root, self._embedding.name, dataset.name)
        dist_key = dataset.get_dist_key(mode)
        try:
            dist_key.index("-")
            gen_dir = join(dataset_dir, "_composites", dist_key)
        except ValueError:
            gen_dir = join(dataset_dir, dist_key)
        makedirs(gen_dir, exist_ok=True)
        return gen_dir

    def _get_tf_record_folder_pattern(self, dataset, mode):
        folder = self._get_tf_record_folder_name(dataset, mode)
        return join(folder, "*.tfrecord")

    def _get_tf_record_folder_name(self, dataset, mode):
        return join(self._get_gen_dir(dataset, mode), "_" + mode)

    def _get_filtered_vocab_file(self, dataset):
        vocab_file = join(self._get_gen_dir(dataset), "_vocab_file.txt")
        if exists(vocab_file):
            return vocab_file
        return self._write_filtered_vocab_file(dataset)

    def _write_tf_record_files(self, dataset, mode, tf_examples):
        np.random.shuffle(tf_examples)
        num_per_shard = int(
            math.ceil(len(tf_examples) / float(self._num_shards))
        )
        total_shards = int(math.ceil(len(tf_examples) / float(num_per_shard)))
        folder_path = self._get_tf_record_folder_name(dataset, mode)
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
        fetch_dict = {}
        for dataset in self._datasets:
            name = dataset.name + dataset.get_dist_key()
            for mode in ["train", "test"]:
                tf_record_file = self._get_tf_record_folder_name(dataset, mode)
                if not exists(tf_record_file):
                    fetch_dict[name] = fetch_dict.get(name, {})
                    fetch_dict[name][mode] = self._append_fetches(
                        dataset, mode
                    )
        return fetch_dict

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
        values, metadata = self._run_fetches()
        self._write_run_metadata(metadata)
        for dataset in self._datasets:
            name = dataset.name + dataset.get_dist_key()
            for mode in values.get(name, []):
                if mode == "train":
                    labels = dataset.train_dict["labels"]
                else:
                    labels = dataset.test_dict["labels"]
                feature_dict = values[name][mode]
                labels = self.zero_norm_labels(labels)
                features = self._feature_lists_from_dict(feature_dict)
                data_zip = zip(*features, labels)
                tf_examples = [self._make_tf_example(*dz) for dz in data_zip]
                self._write_tf_record_files(dataset, mode, tf_examples)

    @timeit("Generating sparse tensors of tokens", "Sparse tensors generated")
    def _sparse_tensors_from_tokens(self, l_tok, trg_tok, r_tok):
        l_sp = [self.get_tokens_sp_tensor(l) for l in l_tok]
        trg_sp = [self.get_tokens_sp_tensor(t) for t in trg_tok]
        r_sp = [self.get_tokens_sp_tensor(r) for r in r_tok]
        return (l_sp, trg_sp, r_sp)

    @timeit("Building graph with required embedding lookup ops", "Graph built")
    def _append_fetches(self, dataset, mode):
        if mode == "train":
            data = dataset.train_dict
        else:
            data = dataset.test_dict

        tokens = self.tokens_from_dict(data)
        self._write_tokens_file(dataset, mode, tokens)

        sparse_tokens = self._sparse_tensors_from_tokens(*tokens)

        vocab_file = self._get_filtered_vocab_file(dataset)
        ids_table = self.index_lookup_table(vocab_file)

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
    def _run_fetches(self):
        if tf.executing_eagerly():
            raise ValueError("Eager execution is not supported.")
        with tf.Session() as sess:
            sess.run(tf.tables_initializer())
            run_opts = tf.RunOptions(
                trace_level=tf.RunOptions.FULL_TRACE  # pylint: disable=E1101
            )
            run_metadata = tf.RunMetadata()
            values = sess.run(
                self.__fetch_dict, options=run_opts, run_metadata=run_metadata
            )

        return values, run_metadata

    @timeit("Exporting lookup table vocabulary file", "Vocab file exported")
    def _write_filtered_vocab_file(self, dataset):
        vocab = self._embedding.vocab
        vocab_set = set(vocab)
        corpus = ["<PAD>", "<OOV>"] + dataset.corpus
        corpus_set = set(c.lower() for c in corpus)

        not_oov_set = set.intersection(vocab_set, corpus_set)

        filtered = [(w, vocab.index(w)) for w in not_oov_set]

        vocab_file_path = join(self._get_gen_dir(dataset), "_vocab_file.txt")
        with open(vocab_file_path, "w") as vocab_file:
            for (word, index) in filtered:
                vocab_file.write("{0}\t{1}\n".format(word, index))

        return vocab_file_path

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
    def _write_tokens_file(self, dataset, mode, tokens):
        file_name = "_" + mode + "_tokens.csv"
        file_path = join(self._get_gen_dir(dataset, mode), file_name)
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
