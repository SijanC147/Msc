import numpy as np
import tensorflow as tf
from os.path import normpath, basename, splitext, dirname
from tsaplay.utils import tokenize_phrase, default_oov

import tsaplay.embeddings._constants as EMBEDDINGS


class Embedding:
    def __init__(self, path, oov=None):
        self.path = path
        self.oov = oov

    @property
    def path(self):
        return self.__path

    @property
    def oov(self):
        return self.__oov

    @property
    def version(self):
        file_name = splitext(basename(normpath(self.path)))[0]
        return file_name.replace("partial_", "")

    @property
    def name(self):
        return basename(normpath(dirname(self.path)))

    @property
    def dictionary(self):
        if self.__dictionary is None:
            self.dictionary = self._load_embedding(path=self.path)
        return self.__dictionary

    @property
    def dim_size(self):
        self.__dim_size = len(next(iter(self.dictionary.values())))
        return self.__dim_size

    @property
    def vocab_size(self):
        self.__vocab_size = len(self.dictionary)
        return self.__vocab_size

    @property
    def initializer(self):
        shape = (self.vocab_size, self.dim_size)

        def _init(shape=shape, dtype=tf.float32, partition_info=None):
            return np.asarray([self.dictionary[w] for w in [*self.dictionary]])

        self.__initializer = _init
        return self.__initializer

    @path.setter
    def path(self, path):
        try:
            path_changed = self.__path != path
        except:
            path_changed = True

        if path_changed:
            self._reset(path)

    @oov.setter
    def oov(self, oov):
        if oov is None:
            self.__oov = lambda dim_size: default_oov(dim_size)
        else:
            self.__oov = lambda dim_size: oov(dim_size)
        self.dictionary["<OOV>"] = self.__oov(dim_size=self.dim_size)

    @dictionary.setter
    def dictionary(self, dictionary):
        if dictionary is not None:
            dim_size = len(next(iter(dictionary.values())))
            dictionary = {**self._get_flags(dim_size), **dictionary}

        self.__dictionary = dictionary

    def filter_on_vocab(self, vocab):
        vocab_lower = [word.lower() for word in vocab]
        filtered_dict = {
            word: self.dictionary[word]
            for word in vocab_lower
            if word in [*self.dictionary]
        }
        self.dictionary = filtered_dict

    def get_index_ids(self, phrase):
        phrase = phrase.decode() if isinstance(phrase, bytes) else phrase
        phrase = str(phrase) if not isinstance(phrase, str) else phrase
        words = tokenize_phrase(phrase.lower())
        vocab = [*self.dictionary]
        ids = [
            vocab.index(word) if word in vocab else vocab.index("<OOV>")
            for word in words
        ]
        return ids

    def _reset(self, path):
        self.__path = path
        self.dictionary = None

    def _load_embedding(self, path):
        dictionary = {}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                values = line.strip().split()
                word = values[0]
                embedding = np.asarray(values[1:], dtype="float32")
                dictionary[word] = embedding
        return dictionary

    def _get_flags(self, dim_size):
        flags = {
            "<PAD>": np.zeros(shape=dim_size),
            "<OOV>": self.oov(dim_size=dim_size),
        }
        return flags
