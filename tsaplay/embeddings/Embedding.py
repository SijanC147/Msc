import numpy as np
import gensim.downloader as gensim_data
from tensorflow import float32 as tf_flt32
from os import makedirs
from os.path import join, normpath, basename, splitext, dirname, exists

import tsaplay.embeddings._constants as EMBEDDINGS


class Embedding:
    def __init__(self, source, oov=None):
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
    def dictionary(self):
        return self._dictionary

    @property
    def data_dir(self):
        data_dir = join(EMBEDDINGS.DATA_PATH, self.name)
        makedirs(data_dir, exist_ok=True)
        return data_dir

    @property
    def vocab(self):
        return [*self.dictionary]

    @property
    def dim_size(self):
        return self._gensim_model.vector_size

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
        vectors = self._gensim_model.vectors
        return np.concatenate([flags, vectors])

    @property
    def initializer(self):
        shape = (self.vocab_size, self.dim_size)

        def _init(shape=shape, dtype=tf_flt32, partition_info=None):
            return self.vectors

        self.__initializer = _init
        return self.__initializer

    @source.setter
    def source(self, new_source):
        try:
            self._source = new_source
            self._gensim_model = gensim_data.load(self._source)
            self._dictionary = {
                **self._get_flags(self.dim_size),
                **self._build_embedding_dictionary(),
            }
            self._export_vocabulary_files()
        except:
            raise ValueError("Invalid source {0}".format(new_source))

    @oov.setter
    def oov(self, oov):
        if oov is None:
            self._oov = lambda size: self._default_oov(size)
        else:
            self._oov = lambda size: oov(size)

    def _default_oov(self, size):
        return np.random.uniform(low=-0.03, high=0.03, size=size)

    def _get_flags(self, dim_size):
        return {
            "<PAD>": np.zeros(shape=dim_size),
            "<OOV>": self.oov(size=dim_size),
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
        words = [*self._gensim_model.vocab]
        vectors = self._gensim_model.vectors
        for (word, vector) in zip(words, vectors):
            dictionary[word] = vector

        return dictionary
