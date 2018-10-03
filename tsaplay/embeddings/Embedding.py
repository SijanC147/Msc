import tensorflow as tf
import numpy as np
import gensim.downloader as gensim_data
from gensim.models import KeyedVectors
from os import makedirs, getcwd
from os.path import join, normpath, basename, splitext, dirname, exists

from tsaplay.utils.decorators import timeit


DATA_PATH = join(getcwd(), "tsaplay", "embeddings", "data")


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
        gen_dir = join(DATA_PATH, self.name)
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
    def initializer(self):
        partition_size = int(self.vocab_size / 6)
        shape = (self.vocab_size, self.dim_size)

        def _init(shape=shape, dtype=tf.float32, partition_info=None):
            part_offset = partition_info.single_offset(shape)
            this_slice = part_offset + partition_size
            return self.vectors[part_offset:this_slice]

        self.__initializer = _init
        return self.__initializer

    @source.setter
    @timeit("Loading embedding model", "Embedding model loaded")
    def source(self, new_source):
        try:
            self._source = new_source
            self._gensim_model = self._load_gensim_model(self._source)
            self._set_vectors()
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
