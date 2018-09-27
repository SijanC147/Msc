import tensorflow as tf
from os.path import join, exists
from os import getcwd, makedirs
from tensorflow.train import BytesList, Feature, Features, Example, Int64List
from tensorflow.python_io import TFRecordWriter

from tsaplay.utils._nlp import tokenize_phrase, inspect_dist, re_dist
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
            "embedding_initializer": self._embedding.initializer,
            "vocab_size": self._embedding.vocab_size,
            "embedding_dim": self._embedding.dim_size,
            "vocab_file_path": self._embedding.vocab_file_path,
        }

    def get_features(self, mode):
        if not exists(self._tfrecord_file(mode)):
            self._export_tf_records(mode)
        features, labels = self._parse_tf_records_file(mode)

        if mode == "train" and self.train_dist is not None:
            features, labels = re_dist(features, labels, self.train_dist)
        if mode == "test" and self.test_dist is not None:
            features, labels = re_dist(features, labels, self.test_dist)

        stats = inspect_dist(features, labels)

        return features, labels, stats

    def _parse_tf_records_file(self, mode):
        feature_spec = {
            "sentence": tf.FixedLenFeature(dtype=tf.string, shape=[]),
            "left_lit": tf.FixedLenFeature(dtype=tf.string, shape=[]),
            "target_lit": tf.FixedLenFeature(dtype=tf.string, shape=[]),
            "right_lit": tf.FixedLenFeature(dtype=tf.string, shape=[]),
            "left_tok": tf.FixedLenFeature(dtype=tf.string, shape=[]),
            "target_tok": tf.FixedLenFeature(dtype=tf.string, shape=[]),
            "right_tok": tf.FixedLenFeature(dtype=tf.string, shape=[]),
            "left_ids": tf.VarLenFeature(dtype=tf.int64),
            "target_ids": tf.VarLenFeature(dtype=tf.int64),
            "right_ids": tf.VarLenFeature(dtype=tf.int64),
            "labels": tf.FixedLenFeature(dtype=tf.int64, shape=[]),
        }

        dataset = tf.data.TFRecordDataset(self._tfrecord_file(mode))
        dataset = dataset.map(
            lambda example: tf.parse_example([example], feature_spec)
        )
        iterator = dataset.make_one_shot_iterator()
        next_example = iterator.get_next()

        features = {
            "sentence": [],
            "left": [],
            "target": [],
            "right": [],
            "mappings": {"left": [], "target": [], "right": []},
        }
        labels = []

        sess = tf.Session()
        while True:
            try:
                feature = sess.run(next_example)

                left_ids_op = tf.sparse_tensor_to_dense(feature["left_ids"])
                target_ids_op = tf.sparse_tensor_to_dense(
                    feature["target_ids"]
                )
                right_ids_op = tf.sparse_tensor_to_dense(feature["right_ids"])

                sentence = feature["sentence"][0].decode("utf-8")
                left_lit = feature["left_lit"][0].decode("utf-8")
                target_lit = feature["target_lit"][0].decode("utf-8")
                right_lit = feature["right_lit"][0].decode("utf-8")

                left_ids = left_ids_op.eval(session=sess)[0].tolist()
                target_ids = target_ids_op.eval(session=sess)[0].tolist()
                right_ids = right_ids_op.eval(session=sess)[0].tolist()

                features["sentence"].append(sentence)
                features["left"].append(left_lit)
                features["target"].append(target_lit)
                features["right"].append(right_lit)
                features["mappings"]["left"].append(left_ids)
                features["mappings"]["target"].append(target_ids)
                features["mappings"]["right"].append(right_ids)
                labels.append(feature["labels"][0])
            except tf.errors.OutOfRangeError:
                break

        sess.close()
        tf.reset_default_graph()

        return features, labels

    def _export_tf_records(self, mode):
        if mode == "train":
            dictionary = self._dataset.train_dict
        else:
            dictionary = self._dataset.test_dict

        partition_dict = self._partition_dictionary_sentences(dictionary)
        tf_encoded_dict = self._tf_encode_dictionary(partition_dict)
        ids_dict = self._convert_to_id_mappings(tf_encoded_dict)

        ids_dict_zip = zip(
            ids_dict["sentence"],
            ids_dict["left_lit"],
            ids_dict["target_lit"],
            ids_dict["right_lit"],
            ids_dict["left_tok"],
            ids_dict["target_tok"],
            ids_dict["right_tok"],
            ids_dict["left_ids"],
            ids_dict["target_ids"],
            ids_dict["right_ids"],
            ids_dict["labels"],
        )

        tf_examples = []

        for (
            sentence,
            left_lit,
            target_lit,
            right_lit,
            left_tok,
            target_tok,
            right_tok,
            left_ids,
            target_ids,
            right_ids,
            label,
        ) in ids_dict_zip:
            features = Features(
                feature={
                    "sentence": Feature(
                        bytes_list=BytesList(value=[sentence.encode()])
                    ),
                    "left_lit": Feature(
                        bytes_list=BytesList(value=[left_lit.encode()])
                    ),
                    "target_lit": Feature(
                        bytes_list=BytesList(value=[target_lit.encode()])
                    ),
                    "right_lit": Feature(
                        bytes_list=BytesList(value=[right_lit.encode()])
                    ),
                    "left_tok": Feature(
                        bytes_list=BytesList(value=[left_tok])
                    ),
                    "target_tok": Feature(
                        bytes_list=BytesList(value=[target_tok])
                    ),
                    "right_tok": Feature(
                        bytes_list=BytesList(value=[right_tok])
                    ),
                    "left_ids": Feature(int64_list=Int64List(value=left_ids)),
                    "target_ids": Feature(
                        int64_list=Int64List(value=target_ids)
                    ),
                    "right_ids": Feature(
                        int64_list=Int64List(value=right_ids)
                    ),
                    "labels": Feature(int64_list=Int64List(value=[label])),
                }
            )
            tf_example = Example(features=features)
            tf_examples.append(tf_example.SerializeToString())

        with TFRecordWriter(self._tfrecord_file(mode)) as tf_writer:
            for serialized_example in tf_examples:
                tf_writer.write(serialized_example)

    def _partition_dictionary_sentences(self, dictionary):
        new_dict = {
            "sentence": [s.strip() for s in dictionary["sentences"]],
            "left_lit": [],
            "target_lit": [t.strip() for t in dictionary["targets"]],
            "right_lit": [],
            "labels": dictionary["labels"],
        }
        dict_zip = zip(
            dictionary["sentences"],
            dictionary["targets"],
            dictionary["offsets"],
        )

        for (sentence, target, offset) in dict_zip:
            left_lit = sentence[:offset].strip()
            r_offset = offset + len(target.strip())
            right_lit = sentence[r_offset:].strip()

            new_dict["left_lit"].append(left_lit)
            new_dict["right_lit"].append(right_lit)

        return new_dict

    def _tf_encode_dictionary(self, dictionary):
        new_dict = {
            **dictionary,
            "left_tok": [],
            "target_tok": [],
            "right_tok": [],
        }
        dict_zip = zip(
            dictionary["left_lit"],
            dictionary["target_lit"],
            dictionary["right_lit"],
        )

        for (left, target, right) in dict_zip:
            new_dict["left_tok"].append(self._tf_encode_string(left))
            new_dict["target_tok"].append(self._tf_encode_string(target))
            new_dict["right_tok"].append(self._tf_encode_string(right))

        return new_dict

    def _convert_to_id_mappings(self, dictionary):
        new_dict = {
            **dictionary,
            "left_ids": [],
            "target_ids": [],
            "right_ids": [],
        }
        tf_dict_zip = zip(
            dictionary["left_tok"],
            dictionary["target_tok"],
            dictionary["right_tok"],
        )

        ids_table = tf.contrib.lookup.index_table_from_file(
            vocabulary_file=self._embedding.vocab_file_path, default_value=1
        )

        with tf.Session():
            tf.tables_initializer().run()

            for (left, target, right) in tf_dict_zip:
                left_ids = self._tf_lookup_ids(ids_table, left)
                target_ids = self._tf_lookup_ids(ids_table, target)
                right_ids = self._tf_lookup_ids(ids_table, right)

                new_dict["left_ids"].append(left_ids.eval()[0])
                new_dict["target_ids"].append(target_ids.eval()[0])
                new_dict["right_ids"].append(right_ids.eval()[0])

        tf.reset_default_graph()

        return new_dict

    def _tf_lookup_ids(self, table, tf_encoded_string):
        string_tensor = tf.constant([tf_encoded_string], dtype=tf.string)
        token_sp_tensor = tf.string_split(string_tensor, delimiter="<SEP>")
        ids_sp_tensor = table.lookup(token_sp_tensor)
        ids_tensor = tf.sparse_tensor_to_dense(ids_sp_tensor)

        return ids_tensor

    def _tf_encode_string(self, string):
        tokenized_string_list = tokenize_phrase(string, lower=True)
        tokenized_string = "<SEP>".join(tokenized_string_list)
        encoded_tokenized_string = tokenized_string.encode()

        return encoded_tokenized_string
