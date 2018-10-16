from os import makedirs
from os.path import join, dirname, exists, basename, normpath
from math import sqrt, floor
import tensorflow as tf
import numpy as np
import gensim.downloader as gensim_data
from gensim.models import KeyedVectors
from tsaplay.utils.decorators import timeit
from tsaplay.constants import EMBEDDING_DATA_PATH

FASTTEXT_WIKI_300 = "fasttext-wiki-news-subwords-300"
GLOVE_TWITTER_25 = "glove-twitter-25"
GLOVE_TWITTER_50 = "glove-twitter-50"
GLOVE_TWITTER_100 = "glove-twitter-100"
GLOVE_TWITTER_200 = "glove-twitter-200"
GLOVE_WIKI_GIGA_50 = "glove-wiki-gigaword-50"
GLOVE_WIKI_GIGA_100 = "glove-wiki-gigaword-100"
GLOVE_WIKI_GIGA_200 = "glove-wiki-gigaword-200"
GLOVE_WIKI_GIGA_300 = "glove-wiki-gigaword-300"
GLOVE_COMMON42_300 = "glove-cc42-300"
GLOVE_COMMON840_300 = "glove-cc840-300"
W2V_GOOGLE_300 = "word2vec-google-news-300"
W2V_RUS_300 = "word2vec-ruscorpora-300"


class Embedding:
    def __init__(self, source, oov=None, data_root=None):
        self._data_root = data_root or EMBEDDING_DATA_PATH
        self.oov = oov
        self.source = source

    @property
    def source(self):
        return self._source

    @property
    def name(self):
        return self._source

    @property
    def oov(self):
        return self._oov

    @property
    def gen_dir(self):
        return self._gen_dir

    @property
    def vocab(self):
        return [*self.flags] + self._gensim_model.index2word

    @property
    def dim_size(self):
        return self._gensim_model.vector_size

    @property
    def vocab_size(self):
        return len(self.vectors)

    @property
    def vocab_file_path(self):
        return join(self.gen_dir, "_vocab.txt")

    @property
    def flags(self):
        return self._flags

    @property
    def vectors(self):
        return self._vectors

    @property
    def num_shards(self):
        return self._num_shards

    @property
    def initializer(self):
        shape = (self.vocab_size, self.dim_size)

        def _init(shape=shape, dtype=tf.float32, partition_info=None):
            return self.vectors

        # return _init
        return lambda: self.vectors

    @property
    def partitioned_initializer(self):
        partition_size = int(self.vocab_size / self.num_shards)
        shape = (self.vocab_size, self.dim_size)

        def _init_part(shape=shape, dtype=tf.float32, partition_info=None):
            part_offset = partition_info.single_offset(shape)
            this_slice = part_offset + partition_size
            return self.vectors[part_offset:this_slice]

        return _init_part

    @source.setter
    @timeit("Loading embedding model", "Embedding model loaded")
    def source(self, new_source):
        try:
            self._source = new_source
            self._gen_dir = join(self._data_root, self.name)
            makedirs(self._gen_dir, exist_ok=True)
            self._gensim_model = self._load_gensim_model(self._source)
            self._set_vectors()
            if not exists(self.vocab_file_path):
                self._export_vocabulary_files()
        except:
            raise ValueError("Invalid source {0}".format(new_source))

    @oov.setter
    def oov(self, oov):
        if oov is None:
            self._oov = lambda size: self.default_oov(size)
        else:
            self._oov = lambda size: oov(size)

    @classmethod
    def default_oov(cls, size):
        return np.random.uniform(low=-0.03, high=0.03, size=size)

    def _set_vectors(self):
        flags = {
            "<PAD>": np.zeros(shape=self.dim_size),
            "<OOV>": self.oov(size=self.dim_size),
        }
        vectors = np.concatenate(
            [[flags["<PAD>"]], [flags["<OOV>"]], self._gensim_model.vectors]
        )
        self._flags = flags
        self._vectors = vectors.astype(np.float32)
        self._num_shards = self._get_smallest_divisor(self.vocab_size)

    def _export_vocabulary_files(self):
        with open(self.vocab_file_path, "w") as f:
            for word in self.vocab:
                if word != "<PAD>":
                    f.write("{0}\n".format(word))
        tsv_file_path = join(self.gen_dir, "_vocab.tsv")
        with open(tsv_file_path, "w") as f:
            for word in self.vocab:
                f.write("{0}\n".format(word))

    def _load_gensim_model(self, source):
        save_model_path = join(self.gen_dir, "_gensim_model.bin")
        if exists(save_model_path):
            return KeyedVectors.load(save_model_path)
        gensim_model = gensim_data.load(source)
        gensim_model.save(save_model_path)
        return gensim_model

    @classmethod
    def _get_smallest_divisor(cls, number):
        if number % 2 == 0:
            return 2
        square_root = floor(sqrt(number))
        for i in (3, square_root, 2):
            if number % i == 0:
                return i
            return 1
