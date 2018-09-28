import tensorflow as tf
from os.path import join, exists
from os import getcwd, makedirs
from json import dumps
from tensorflow.train import BytesList, Feature, Features, Example, Int64List
from tensorflow.python_io import TFRecordWriter

from tsaplay.utils._nlp import tokenize_phrase, inspect_dist, re_dist
from tsaplay.utils._data import parse_tf_example
from tsaplay.embeddings.Embedding import Embedding
from tsaplay.embeddings.PartialEmbedding import PartialEmbedding
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
        makedirs(self.gen_dir, exist_ok=True)

    @property
    def name(self):
        if isinstance(self._embedding, PartialEmbedding):
            return self._embedding.name
        return "_".join([self._embedding.name, self._dataset.name])

    @property
    def gen_dir(self):
        if isinstance(self._embedding, PartialEmbedding):
            return join(FEATURES.DATA_PATH, self._embedding.name)
        return join(
            FEATURES.DATA_PATH, self._embedding.name, self._dataset.name
        )

    @property
    def embedding_params(self):
        return {
            "vocab_size": self._embedding.vocab_size,
            "vocab_file_path": self._embedding.vocab_file_path,
            "embedding_dim": self._embedding.dim_size,
            "embedding_initializer": self._embedding.initializer,
        }

    def get_tfrecord(self, mode):
        if not exists(self._tfrecord_file(mode)):
            self._export_tf_records(mode)
        return self._tfrecord_file(mode)

    def _export_tf_records(self, mode):
        if mode == "train":
            data = self._dataset.train_dict
        else:
            data = self._dataset.test_dict

        left_ctxts, targets, right_ctxts = self._partition_sentences(
            sentences=data["sentences"],
            targets=data["targets"],
            offsets=data["offsets"],
        )
        left_enc = [self._tf_encode_string(l) for l in left_ctxts]
        target_enc = [self._tf_encode_string(t) for t in targets]
        right_enc = [self._tf_encode_string(r) for r in right_ctxts]

        left_sp = [self._get_tokens_sp_tensor(l) for l in left_enc]
        target_sp = [self._get_tokens_sp_tensor(t) for t in target_enc]
        right_sp = [self._get_tokens_sp_tensor(r) for r in right_enc]

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

    def _partition_sentences(self, sentences, targets, offsets):
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
                left_ids_op = self._tf_lookup_ids(ids_table, l)
                target_ids_op = self._tf_lookup_ids(ids_table, t)
                right_ids_op = self._tf_lookup_ids(ids_table, r)

                left_ids.append(left_ids_op.eval()[0])
                target_ids.append(target_ids_op.eval()[0])
                right_ids.append(right_ids_op.eval()[0])

                left_op = tf.sparse_tensor_to_dense(l, default_value=b"")
                target_op = tf.sparse_tensor_to_dense(t, default_value=b"")
                right_op = tf.sparse_tensor_to_dense(r, default_value=b"")

                left.append(left_op.eval()[0])
                target.append(target_op.eval()[0])
                right.append(right_op.eval()[0])

        tf.reset_default_graph()

        return left, left_ids, target, target_ids, right, right_ids

    def _get_tokens_sp_tensor(self, tf_encoded_string, delimiter="<SEP>"):
        string_tensor = tf.constant([tf_encoded_string], dtype=tf.string)
        tokens_sp_tensor = tf.string_split(string_tensor, delimiter)

        return tokens_sp_tensor

    def _tf_lookup_ids(self, table, tokens_sp_tensor):
        ids_sp_tensor = table.lookup(tokens_sp_tensor)
        ids_tensor = tf.sparse_tensor_to_dense(ids_sp_tensor)

        return ids_tensor

    def _tf_encode_string(self, string):
        tokenized_string_list = tokenize_phrase(string, lower=True)
        tokenized_string = "<SEP>".join(tokenized_string_list)
        encoded_tokenized_string = tokenized_string.encode()

        return encoded_tokenized_string
