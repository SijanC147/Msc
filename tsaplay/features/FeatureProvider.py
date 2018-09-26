import tensorflow as tf
from os.path import join, exists
from os import getcwd, makedirs
from tensorflow.train import BytesList, Feature, Features, Example, Int64List
from tensorflow.python_io import TFRecordWriter

from tsaplay.utils._nlp import tokenize_phrase
from tsaplay.embeddings.Embedding import Embedding
from tsaplay.embeddings.PartialEmbedding import PartialEmbedding
import tsaplay.features._constants as FEATURES


class FeatureProvider:
    def __init__(self, dataset, embedding):
        self._dataset = dataset
        self._embedding = embedding
        self._tfrecord_file = lambda mode: join(
            self.gen_dir, "_" + mode + ".tfrecord"
        )
        makedirs(self.gen_dir, exist_ok=True)

    @property
    def gen_dir(self):
        if isinstance(self._embedding, PartialEmbedding):
            return join(FEATURES.DATA_PATH, self._embedding.name)
        return join(
            FEATURES.DATA_PATH, self._embedding.name, self._dataset.name
        )

    def get_features(self, mode):
        if not exists(self._tfrecord_file(mode)):
            self._export_tf_records(mode)

        return self._parse_tf_records_file(mode)

    def _parse_tf_records_file(self, mode):
        feature_spec = {
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

        features = {"left_ids": [], "target_ids": [], "right_ids": []}
        labels = []

        sess = tf.Session()
        while True:
            try:
                feature = sess.run(next_example)
                left = tf.sparse_tensor_to_dense(feature["left_ids"])
                target = tf.sparse_tensor_to_dense(feature["target_ids"])
                right = tf.sparse_tensor_to_dense(feature["right_ids"])

                features["left_ids"].append(
                    left.eval(session=sess)[0].tolist()
                )
                features["target_ids"].append(
                    target.eval(session=sess)[0].tolist()
                )
                features["right_ids"].append(
                    right.eval(session=sess)[0].tolist()
                )
                labels.append(feature["labels"][0])
            except tf.errors.OutOfRangeError:
                break
        sess.close()
        tf.reset_default_graph()

        return features, labels

    def _convert_to_id_mappings(self, dictionary):
        tf_dict = self._partition_dictionary_sentences(dictionary)
        tf_dict_zip = zip(
            tf_dict["left_ctxts"], tf_dict["targets"], tf_dict["right_ctxts"]
        )
        new_dict = {
            "left_ids": [],
            "target_ids": [],
            "right_ids": [],
            "labels": tf_dict["labels"],
        }

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

    def _tf_encoded_tokens(self, string):
        tokenized_string_list = tokenize_phrase(string, lower=True)
        tokenized_string = "<SEP>".join(tokenized_string_list)
        encoded_tokenized_string = tokenized_string.encode()

        return encoded_tokenized_string

    def _partition_dictionary_sentences(self, dictionary):
        new_dict = {
            "left_ctxts": [],
            "targets": [],
            "right_ctxts": [],
            "labels": dictionary["labels"],
        }
        dict_zip = zip(
            dictionary["sentences"],
            dictionary["targets"],
            dictionary["offsets"],
            dictionary["labels"],
        )

        for (sentence, target, offset, _) in dict_zip:
            left_ctxt = self._tf_encoded_tokens(sentence[:offset].strip())
            target = self._tf_encoded_tokens(target)
            r_offset = offset + len(target.strip())
            right_ctxt = self._tf_encoded_tokens(sentence[r_offset:].strip())

            new_dict["left_ctxts"].append(left_ctxt)
            new_dict["targets"].append(target)
            new_dict["right_ctxts"].append(right_ctxt)

        return new_dict

    def _export_tf_records(self, mode):
        if mode == "train":
            dictionary = self._dataset.train_dict
        else:
            dictionary = self._dataset.test_dict

        ids_dict = self._convert_to_id_mappings(dictionary)
        ids_dict_zip = zip(
            ids_dict["left_ids"],
            ids_dict["target_ids"],
            ids_dict["right_ids"],
            ids_dict["labels"],
        )
        tf_examples = []

        for (left, target, right, label) in ids_dict_zip:
            features = Features(
                feature={
                    "left_ids": Feature(int64_list=Int64List(value=left)),
                    "target_ids": Feature(int64_list=Int64List(value=target)),
                    "right_ids": Feature(int64_list=Int64List(value=right)),
                    "labels": Feature(int64_list=Int64List(value=[label])),
                }
            )
            tf_example = Example(features=features)
            tf_examples.append(tf_example.SerializeToString())

        with TFRecordWriter(self._tfrecord_file(mode)) as tf_writer:
            for serialized_example in tf_examples:
                tf_writer.write(serialized_example)
