import numpy as np
import tensorflow as tf
import time
import gensim.downloader as gensim_data
from os import makedirs, getcwd
from os.path import join, normpath, basename, splitext, dirname, exists
from tsaplay.utils._nlp import tokenize_phrase, default_oov
from tsaplay.utils._io import gprint

import tsaplay.embeddings._constants as EMBEDDINGS


class Embedding:
    def __init__(self, source, oov=None):
        self.source = source
        self.oov = oov

    @property
    def source(self):
        return self.__source

    @property
    def name(self):
        return self.__source

    @property
    def oov(self):
        return self.__oov

    @property
    def dictionary(self):
        return self.__dictionary

    @property
    def data_dir(self):
        return join(EMBEDDINGS.DATA_PATH, self.source)

    @property
    def vocab(self):
        return [*self.dictionary]

    @property
    def dim_size(self):
        return self.__gensim_model.vector_size

    @property
    def vocab_size(self):
        return len(self.dictionary)

    @property
    def vocab_file_path(self):
        return join(self.data_dir, "_vocab.txt")

    @property
    def vectors(self):
        flags = np.asarray(
            [self.dictionary["<PAD>"], self.dictionary["<OOV>"]]
        )
        vectors = self.__gensim_model.vectors
        return np.concatenate([flags, vectors])

    @property
    def initializer(self):
        shape = (self.vocab_size, self.dim_size)

        def _init(shape=shape, dtype=tf.float32, partition_info=None):
            return self.vectors

        self.__initializer = _init
        return self.__initializer

    @source.setter
    def source(self, new_source):
        try:
            self.__source = new_source
            self.__gensim_model = gensim_data.load(self.__source)
            self.__dictionary = {
                **self._get_flags(self.dim_size),
                **self._build_embedding_dictionary(),
            }
            self._export_vocabulary_files()
        except:
            raise ValueError("Invalid source {0}".format(new_source))

    @oov.setter
    def oov(self, oov):
        if oov is None:
            self.__oov = lambda dim_size: default_oov(dim_size)
        else:
            self.__oov = lambda dim_size: oov(dim_size)

    @gensim_model.setter
    def gensim_model(self, gensim_model):
        self.__gensim_model = gensim_model

    def _get_flags(self, dim_size):
        return {
            "<PAD>": np.zeros(shape=dim_size),
            "<OOV>": self.oov(dim_size=dim_size),
        }

    def _export_vocabulary_files(self):
        makedirs(dirname(self.vocab_file_path), exist_ok=True)
        with open(self.vocab_file_path, "w") as f:
            for word in [*self.dictionary]:
                if word != "<PAD>":
                    f.write("{0}\n".format(word))
        tsv_file_path = join(self.data_dir, "_vocab.tsv")
        with open(tsv_file_path, "w") as f:
            for word in [*self.dictionary]:
                f.write("{0}\n".format(word))

    def _build_embedding_dictionary(self):
        dictionary = {}
        words = [*self.__gensim_model.vocab]
        vectors = self.__gensim_model.vectors
        for (word, vector) in zip(words, vectors):
            dictionary[word] = vector

        return dictionary
