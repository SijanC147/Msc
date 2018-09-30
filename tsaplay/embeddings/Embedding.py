import numpy as np
import gensim.downloader as gensim_data
from gensim.models import KeyedVectors
from tensorflow import float32 as tf_flt32
from os import makedirs
from os.path import join, normpath, basename, splitext, dirname, exists

import tsaplay.embeddings._constants as EMBEDDINGS
from tsaplay.models._decorators import timeit


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
    def gen_dir(self):
        gen_dir = join(EMBEDDINGS.DATA_PATH, self.name)
        makedirs(gen_dir, exist_ok=True)
        return gen_dir

    @property
    def vocab(self):
        return [*self.flags] + self._gensim_model.index2word

    @property
    def dim_size(self):
        return self._gensim_model.vector_size

    @property
    def vocab_size(self):
        return len(self.vocab)

    @property
    def vocab_file_path(self):
        return join(self.gen_dir, "_vocab.txt")

    @property
    def flags(self):
        return {
            "<PAD>": np.zeros(shape=self.dim_size),
            "<OOV>": self.oov(size=self.dim_size),
        }

    @property
    def vectors(self):
        return np.concatenate(
            [
                [self.flags["<PAD>"]],
                [self.flags["<OOV>"]],
                self._gensim_model.vectors,
            ]
        )

    @property
    def initializer(self):
        shape = (self.vocab_size, self.dim_size)

        def _init(shape=shape, dtype=tf_flt32, partition_info=None):
            return self.vectors

        self.__initializer = _init
        return self.__initializer

    @source.setter
    @timeit
    def source(self, new_source):
        try:
            self._source = new_source
            self._gensim_model = self._load_gensim_model(self._source)
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
        else:
            gensim_model = gensim_data.load(source)
            gensim_model.save(save_model_path)
            return gensim_model
