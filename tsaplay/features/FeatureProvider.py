import tensorflow as tf
from os.path import join, exists, isfile
from os import getcwd, makedirs, listdir
from json import dumps
from tensorflow.train import BytesList, Feature, Features, Example, Int64List
from tensorflow.python_io import TFRecordWriter

from tsaplay.utils._nlp import tokenize_phrase, inspect_dist
from tsaplay.utils._data import parse_tf_example
from tsaplay.utils._io import gprint
from tsaplay.embeddings.Embedding import Embedding
from tsaplay.embeddings.PartialEmbedding import PartialEmbedding
from tsaplay.datasets.CompoundDataset import CompoundDataset
import tsaplay.features._constants as FEATURES


class FeatureProvider:
    def __init__(self, dataset, embedding, train_dist=None, test_dist=None):
        self.train_dist = train_dist
        self.test_dist = test_dist
        self._dataset = dataset
        self._embedding = embedding
        self._tfrecord_file = lambda mode: join(
            self.gen_dir, "_" + mode + ".tfrecord"
        )

    @property
    def name(self):
        if isinstance(self._embedding, PartialEmbedding):
            return self._embedding.name
        return "_".join([self._embedding.name, self._dataset.name])

    @property
    def gen_dir(self):
        if isinstance(self._embedding, PartialEmbedding):
            gen_dir = join(FEATURES.DATA_PATH, self._embedding.name)
        else:
            gen_dir = join(
                FEATURES.DATA_PATH, self._embedding.name, self._dataset.name
            )
        makedirs(gen_dir, exist_ok=True)
        return gen_dir

    @property
    def embedding_params(self):
        return {
            "vocab_size": self._embedding.vocab_size,
            "vocab_file_path": self._embedding.vocab_file_path,
            "embedding_dim": self._embedding.dim_size,
            "embedding_initializer": self._embedding.initializer,
        }

    def get_tfrecord(self, mode):
        compound_dataset = isinstance(self._dataset, CompoundDataset)
        partial_embedding = isinstance(self._embedding, PartialEmbedding)
        if compound_dataset and not partial_embedding:
            tf_record_files = []
            constituent_datasets = self._dataset.datasets
            for dataset in constituent_datasets:
                self._dataset = dataset
                if not exists(self._tfrecord_file(mode)):
                    self._export_tf_records(mode)
                tf_record_files.append(self._tfrecord_file(mode))
            return tf_record_files
        else:
            if not exists(self._tfrecord_file(mode)):
                self._export_tf_records(mode)
            return self._tfrecord_file(mode)

    def _export_tf_records(self, mode):
        if mode == "train":
            data = self._dataset.train_dict
        else:
            data = self._dataset.test_dict

        left_sp, target_sp, right_sp = FeatureProvider.bytes_sp_from_dict(data)

        left, l_ids, target, t_ids, right, r_ids = self._get_ids_bytes_lists(
            left_sp=left_sp, target_sp=target_sp, right_sp=right_sp
        )
        zero_norm_labels = [
            [l + abs(min(data["labels"]))] for l in data["labels"]
        ]
        data_zip = zip(
            left, l_ids, target, t_ids, right, r_ids, zero_norm_labels
        )

        dataset_stats = inspect_dist(left, target, right, data["labels"])

        tf_examples = []
        for (left, l_ids, target, t_ids, right, r_ids, label) in data_zip:
            features = Features(
                feature={
                    "left": Feature(bytes_list=BytesList(value=left)),
                    "target": Feature(bytes_list=BytesList(value=target)),
                    "right": Feature(bytes_list=BytesList(value=right)),
                    "left_ids": Feature(int64_list=Int64List(value=l_ids)),
                    "target_ids": Feature(int64_list=Int64List(value=t_ids)),
                    "right_ids": Feature(int64_list=Int64List(value=r_ids)),
                    "labels": Feature(int64_list=Int64List(value=label)),
                }
            )
            tf_example = Example(features=features)
            tf_examples.append(tf_example.SerializeToString())

        with TFRecordWriter(self._tfrecord_file(mode)) as tf_writer:
            for serialized_example in tf_examples:
                tf_writer.write(serialized_example)

        with open(join(self.gen_dir, "_" + mode + ".json"), "w") as f:
            f.write(dumps(dataset_stats))

    @classmethod
    def bytes_sp_from_dict(cls, dictionary):
        offsets = dictionary.get("offsets", [])
        if len(offsets) == 0:
            offsets = FeatureProvider.get_target_offset_array(dictionary)
        dictionary = {**dictionary, "offsets": offsets}
        l_ctxts, trgs, r_ctxts = FeatureProvider.partition_sentences(
            sentences=dictionary["sentences"],
            targets=dictionary["targets"],
            offsets=dictionary["offsets"],
        )
        l_enc = [FeatureProvider.tf_encode_string(l) for l in l_ctxts]
        trg_enc = [FeatureProvider.tf_encode_string(t) for t in trgs]
        r_enc = [FeatureProvider.tf_encode_string(r) for r in r_ctxts]

        l_sp = [FeatureProvider.get_tokens_sp_tensor(l) for l in l_enc]
        trg_sp = [FeatureProvider.get_tokens_sp_tensor(t) for t in trg_enc]
        r_sp = [FeatureProvider.get_tokens_sp_tensor(r) for r in r_enc]
        return l_sp, trg_sp, r_sp

    @classmethod
    def get_target_offset_array(cls, dictionary):
        offsets = []
        for (s, t) in zip(dictionary["sentences"], dictionary["targets"]):
            offsets.append(s.lower().find(t.lower()))
        return offsets

    @classmethod
    def tf_encode_string(cls, string):
        tokenized_string_list = tokenize_phrase(string, lower=True)
        tokenized_string = "<SEP>".join(tokenized_string_list)
        encoded_tokenized_string = tokenized_string.encode()
        return encoded_tokenized_string

    @classmethod
    def get_tokens_sp_tensor(cls, tf_encoded_string, delimiter="<SEP>"):
        string_tensor = tf.constant([tf_encoded_string], dtype=tf.string)
        tokens_sp_tensor = tf.string_split(string_tensor, delimiter)
        return tokens_sp_tensor

    @classmethod
    def tf_lookup_string_to_ids(cls, table, tokens_sp_tensor):
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

    def _get_ids_bytes_lists(self, left_sp, target_sp, right_sp):
        left, target, right = [], [], []
        left_ids, target_ids, right_ids = [], [], []
        ids_table = tf.contrib.lookup.index_table_from_file(
            vocabulary_file=self._embedding.vocab_file_path, default_value=1
        )

        with tf.Session():
            tf.tables_initializer().run()

            for (l, t, r) in zip(left_sp, target_sp, right_sp):
                left_ids_op = FeatureProvider.tf_lookup_string_to_ids(
                    ids_table, l
                )
                target_ids_op = FeatureProvider.tf_lookup_string_to_ids(
                    ids_table, t
                )
                right_ids_op = FeatureProvider.tf_lookup_string_to_ids(
                    ids_table, r
                )

                left_ids.append(left_ids_op.eval()[0].tolist())
                target_ids.append(target_ids_op.eval()[0].tolist())
                right_ids.append(right_ids_op.eval()[0].tolist())

                left_op = tf.sparse_tensor_to_dense(l, default_value=b"")
                target_op = tf.sparse_tensor_to_dense(t, default_value=b"")
                right_op = tf.sparse_tensor_to_dense(r, default_value=b"")

                left.append(left_op.eval()[0].tolist())
                target.append(target_op.eval()[0].tolist())
                right.append(right_op.eval()[0].tolist())

        return left, left_ids, target, target_ids, right, right_ids

