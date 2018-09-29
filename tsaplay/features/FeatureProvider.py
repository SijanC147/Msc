import tensorflow as tf
from tqdm import tqdm
from multiprocessing.dummy import Pool as ThreadPool
from os.path import join, exists, isfile
from os import getcwd, makedirs, listdir
from json import dumps
from tensorflow.train import BytesList, Feature, Features, Example, Int64List
from tensorflow.python.client.timeline import Timeline  # pylint: disable=E0611
from tensorflow.python_io import TFRecordWriter

from tsaplay.models._decorators import timeit
from tsaplay.utils._nlp import tokenize_phrase, inspect_dist, tokenize_phrases
from tsaplay.utils._data import parse_tf_example
from tsaplay.utils._io import gprint
from tsaplay.embeddings.Embedding import Embedding
from tsaplay.embeddings.PartialEmbedding import PartialEmbedding
from tsaplay.datasets.CompoundDataset import CompoundDataset
import tsaplay.features._constants as FEATURES


class FeatureProvider:
    def __init__(self, datasets, embedding, train_dist=None, test_dist=None):
        self.train_dist = train_dist
        self.test_dist = test_dist
        self._embedding = embedding
        self._datasets = CompoundDataset(datasets)
        # self.__tf_train = self._export_tf_record_files(mode="train")
        self.__tf_test = self._export_tf_record_files(mode="test")

    @property
    def name(self):
        if isinstance(self._embedding, PartialEmbedding):
            return self._embedding.name
        return "_".join([self._embedding.name, self._datasets.name])

    @property
    def embedding_params(self):
        return {
            "vocab_size": self._embedding.vocab_size,
            "vocab_file_path": self._embedding.vocab_file_path,
            "embedding_dim": self._embedding.dim_size,
            "embedding_initializer": self._embedding.initializer,
        }

    # @property
    # def train_tfrecords(self):
    #     return self.__tf_train

    @property
    def test_tfrecords(self):
        return self.__tf_test

    def _get_gen_dir(self, dataset):
        if isinstance(self._embedding, PartialEmbedding):
            gen_dir = join(FEATURES.DATA_PATH, self._embedding.name)
        else:
            gen_dir = join(
                FEATURES.DATA_PATH, self._embedding.name, dataset.name
            )
        makedirs(gen_dir, exist_ok=True)
        return gen_dir

    def _get_tf_record_file_name(self, dataset, mode):
        return join(self._get_gen_dir(dataset), "_" + mode + ".tfrecord")

    def _write_tf_record_file(self, dataset, mode, serialized_examples):
        file_name = "_" + mode + ".tfrecord"
        file_path = join(self._get_gen_dir(dataset), file_name)

        with TFRecordWriter(file_path) as tf_writer:
            for serialized_example in serialized_examples:
                tf_writer.write(serialized_example)

        return file_path

    def _export_tf_record_files(self, mode):
        tf_record_files = []
        for dataset in self._datasets.datasets:
            if mode == "train":
                data = dataset.train_dict
            else:
                data = dataset.test_dict

            left_sp, target_sp, right_sp = self.bytes_sp_from_dict(data)

            return
            vocab_file = self._export_filtered_vocab_file(dataset)
            lookup_table = self._get_index_lookup_table(vocab_file)

            mapped_features, run_metadata = self._get_ids_bytes(
                table=lookup_table,
                left_sp=left_sp,
                target_sp=target_sp,
                right_sp=right_sp,
            )
            labels = self.zero_norm_labels(data["labels"])

            self._write_run_metadata(dataset, mode, run_metadata)

            tf_examples = []
            data_zip = zip(*mapped_features, labels)
            for (left, l_ids, target, t_ids, right, r_ids, label) in data_zip:
                feature = {
                    "left": Feature(bytes_list=BytesList(value=left)),
                    "target": Feature(bytes_list=BytesList(value=target)),
                    "right": Feature(bytes_list=BytesList(value=right)),
                    "left_ids": Feature(int64_list=Int64List(value=l_ids)),
                    "target_ids": Feature(int64_list=Int64List(value=t_ids)),
                    "right_ids": Feature(int64_list=Int64List(value=r_ids)),
                    "labels": Feature(int64_list=Int64List(value=[label])),
                }

                tf_example = Example(features=Features(feature=feature))
                tf_examples.append(tf_example.SerializeToString())

            file_path = self._write_tf_record_file(dataset, mode, tf_examples)
            tf_record_files.append(file_path)

        return tf_record_files

    # @classmethod
    # @timeit
    # def bytes_sp_from_dict(cls, dictionary):
    #     offsets = dictionary.get("offsets", [])
    #     if len(offsets) == 0:
    #         offsets = cls.get_target_offset_array(dictionary)
    #     dictionary = {**dictionary, "offsets": offsets}
    #     left, target, right = cls.partition_sentences(
    #         sentences=dictionary["sentences"],
    #         targets=dictionary["targets"],
    #         offsets=dictionary["offsets"],
    #     )
    # l_sp = [
    #     cls.get_tokens_sp_tensor(cls.tf_encode_string(l))
    #     for l in tqdm(left)
    # ]
    # trg_sp = [
    #     cls.get_tokens_sp_tensor(cls.tf_encode_string(t))
    #     for t in tqdm(target)
    # ]
    # r_sp = [
    #     cls.get_tokens_sp_tensor(cls.tf_encode_string(r))
    #     for r in tqdm(right)
    # ]
    # l_sp, trg_sp, r_sp = [], [], []
    # for (l, t, r) in zip(tqdm(left), tqdm(target), tqdm(right)):
    #     l_sp.append(cls.get_tokens_sp_tensor(cls.tf_encode_string(l)))
    #     trg_sp.append(cls.get_tokens_sp_tensor(cls.tf_encode_string(t)))
    #     r_sp.append(cls.get_tokens_sp_tensor(cls.tf_encode_string(r)))

    # return l_sp, trg_sp, r_sp

    @classmethod
    @timeit
    def bytes_sp_from_dict(cls, dictionary):
        offsets = dictionary.get("offsets", [])
        if len(offsets) == 0:
            offsets = cls.get_target_offset_array(dictionary)
        dictionary = {**dictionary, "offsets": offsets}
        l_ctxts, trgs, r_ctxts = cls.partition_sentences(
            sentences=dictionary["sentences"],
            targets=dictionary["targets"],
            offsets=dictionary["offsets"],
        )
        l_enc = ["<SEP>".join(l).encode() for l in tokenize_phrases(l_ctxts)]
        trg_enc = ["<SEP>".join(t).encode() for t in tokenize_phrases(trgs)]
        r_enc = ["<SEP>".join(r).encode() for r in tokenize_phrases(r_ctxts)]

        l_sp = [cls.get_tokens_sp_tensor(l, cls.short_name(l)) for l in l_enc]
        trg_sp = [
            cls.get_tokens_sp_tensor(t, cls.short_name(t)) for t in trg_enc
        ]
        r_sp = [cls.get_tokens_sp_tensor(r, cls.short_name(r)) for r in r_enc]
        cls.print_tensors(l_sp, trg_sp, r_sp)
        return l_sp, trg_sp, r_sp

    # @classmethod
    # @timeit
    # def bytes_sp_from_dict(cls, dictionary):
    #     offsets = dictionary.get("offsets", [])
    #     if len(offsets) == 0:
    #         offsets = cls.get_target_offset_array(dictionary)
    #     dictionary = {**dictionary, "offsets": offsets}
    #     l_ctxts, trgs, r_ctxts = cls.partition_sentences(
    #         sentences=dictionary["sentences"],
    #         targets=dictionary["targets"],
    #         offsets=dictionary["offsets"],
    #     )
    #     l_enc = [cls.tf_encode_string(l) for l in l_ctxts]
    #     trg_enc = [cls.tf_encode_string(t) for t in trgs]
    #     r_enc = [cls.tf_encode_string(r) for r in r_ctxts]

    #     l_sp = [cls.get_tokens_sp_tensor(l, cls.short_name(l)) for l in l_enc]
    #     trg_sp = [
    #         cls.get_tokens_sp_tensor(t, cls.short_name(t)) for t in trg_enc
    #     ]
    #     r_sp = [cls.get_tokens_sp_tensor(r, cls.short_name(r)) for r in r_enc]
    #     cls.print_tensors(l_sp, trg_sp, r_sp)
    #     return l_sp, trg_sp, r_sp

    @classmethod
    def short_name(cls, string, numchar=20):
        name = str(string, "utf-8")
        if len(name) > numchar:
            name = name[:numchar]
        return "".join(filter(str.isalpha, name))

    @classmethod
    def print_tensors(cls, left, target, right):
        template = "{3}:\t {0}\t {1}\t {2}\n"
        for i in range(20):
            gprint(
                template.format(
                    left[i].name, target[i].name, right[i].name, i + 1
                )
            )

    # @classmethod
    # @timeit
    # def bytes_sp_from_dict(cls, dictionary):
    #     offsets = dictionary.get("offsets", [])
    #     if len(offsets) == 0:
    #         offsets = cls.get_target_offset_array(dictionary)
    #     dictionary = {**dictionary, "offsets": offsets}
    #     l_ctxts, trgs, r_ctxts = cls.partition_sentences(
    #         sentences=dictionary["sentences"],
    #         targets=dictionary["targets"],
    #         offsets=dictionary["offsets"],
    #     )
    #     pool = ThreadPool(8)

    #     l_sp = pool.map(cls.get_bytes_sparse, l_ctxts)
    #     trg_sp = pool.map(cls.get_bytes_sparse, trgs)
    #     r_sp = pool.map(cls.get_bytes_sparse, r_ctxts)

    #     cls.print_tensors(l_sp, trg_sp, r_sp)

    #     return l_sp, trg_sp, r_sp

    @classmethod
    def get_bytes_sparse(cls, string):
        enc = cls.tf_encode_string(string)
        return cls.get_tokens_sp_tensor(enc, cls.short_name(enc))

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
    def get_tokens_sp_tensor(cls, tf_enc_string, name=None, delimiter="<SEP>"):
        string_tensor = tf.constant([tf_enc_string], dtype=tf.string)
        tokens_sp_tensor = tf.string_split(string_tensor, delimiter)
        if name is None:
            return tokens_sp_tensor
        else:
            if len(name) > 0:
                return tf.identity(string_tensor, name=name)
            else:
                return tf.identity(string_tensor, name="too_short")

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
        return left_ctxts[:100], _targets[:100], right_ctxts[:100]

    @classmethod
    def zero_norm_labels(cls, labels):
        minimum = abs(min(labels))
        return [l + minimum for l in labels]

    @timeit
    def _export_filtered_vocab_file(self, dataset):
        vocab = self._embedding.vocab
        vocab_set = set(vocab)
        corpus = ["<PAD>", "<OOV>"] + dataset.corpus
        corpus_set = set([c.lower() for c in corpus])

        not_oov_set = set.intersection(vocab_set, corpus_set)

        filtered = [(w, vocab.index(w)) for w in not_oov_set]

        vocab_file = join(self._get_gen_dir(dataset), "_vocab_file.txt")
        with open(vocab_file, "w") as f:
            for (word, index) in filtered:
                f.write("{0}\t{1}\n".format(word, index))

        return vocab_file

    def _write_run_metadata(self, dataset, mode, run_metadata):
        file_name = "_" + mode + "_meta.json"
        file_path = join(self._get_gen_dir(dataset), file_name)

        tl = Timeline(run_metadata.step_stats)  # pylint: disable=E1101
        ctf = tl.generate_chrome_trace_format()
        with open(file_path, "w") as f:
            f.write(ctf)

        return file_path

    def _get_index_lookup_table(self, vocab_file):
        return tf.contrib.lookup.index_table_from_file(
            vocabulary_file=vocab_file,
            key_column_index=0,
            value_column_index=1,
            default_value=1,
            delimiter="\t",
        )

    @timeit
    def _get_ids_bytes(self, table, left_sp, target_sp, right_sp):
        left, target, right = [], [], []
        left_ids, target_ids, right_ids = [], [], []
        left_ids_ops, target_ids_ops, right_ids_ops = [], [], []
        left_ops, target_ops, right_ops = [], [], []

        for (l, t, r) in zip(left_sp, target_sp, right_sp):
            left_ids_ops.append(self.tf_lookup_string_sp(table, l))
            target_ids_ops.append(self.tf_lookup_string_sp(table, t))
            right_ids_ops.append(self.tf_lookup_string_sp(table, r))

            left_ops.append(tf.sparse_tensor_to_dense(l, default_value=b""))
            target_ops.append(tf.sparse_tensor_to_dense(t, default_value=b""))
            right_ops.append(tf.sparse_tensor_to_dense(r, default_value=b""))

        with tf.Session() as sess:
            sess.run(tf.tables_initializer())
            run_opts = tf.RunOptions(
                trace_level=tf.RunOptions.FULL_TRACE  # pylint: disable=E1101
            )
            run_meta = tf.RunMetadata()
            values = sess.run(
                [
                    left_ids_ops,
                    target_ids_ops,
                    right_ids_ops,
                    left_ops,
                    target_ops,
                    right_ops,
                ],
                options=run_opts,
                run_metadata=run_meta,
            )

        left_ids = [l.tolist()[0] for l in values[0]]
        target_ids = [t.tolist()[0] for t in values[1]]
        right_ids = [r.tolist()[0] for r in values[2]]

        left = [l.tolist()[0] for l in values[3]]
        target = [t.tolist()[0] for t in values[4]]
        right = [r.tolist()[0] for r in values[5]]

        return (left, left_ids, target, target_ids, right, right_ids), run_meta
