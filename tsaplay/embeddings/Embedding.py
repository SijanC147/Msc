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
    def oov(self):
        return self.__oov

    @property
    def version(self):
        if self._source_is_path():
            file_name = splitext(basename(normpath(self.source)))[0]
            version = file_name.replace("partial_", "")
        else:
            version = "-".join(self.source.split("-")[1:])
        return version

    @property
    def name(self):
        if self._source_is_path():
            name = basename(normpath(dirname(self.source)))
        else:
            name = self.source.split("-")[0]
        return name

    @property
    def gensim_model(self):
        if not (self._source_is_path()):
            if self.__gensim_model is None:
                self.gensim_model = gensim_data.load(self.source)
        return self.__gensim_model

    @property
    def dictionary(self):
        if self.__dictionary is None:
            if self._source_is_path():
                gprint("Loading embedding from path.")
                self.dictionary = self._load_embedding(path=self.source)
            else:
                gprint("Loading embedding from gensim.")
                self.dictionary = self._get_dict_from_gensim_model(
                    source=self.source
                )
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
        if self._source_is_path():
            np_array = np.asarray(
                [self.dictionary[w] for w in [*self.dictionary]]
            )
        else:
            flags = np.asarray(
                [self.dictionary["<PAD>"], self.dictionary["<OOV>"]]
            )
            np_array_no_flags = self.gensim_model.vectors
            np_array = np.concatenate([flags, np_array_no_flags])

        def _init(shape=shape, dtype=tf.float32, partition_info=None):
            return np_array

        self.__initializer = _init
        return self.__initializer

    @property
    def vocab_file_path(self):
        vocab_file_name = "vocab_" + self.version + ".txt"
        if self._source_is_path():
            vocab_file_path = join(
                dirname(self.source), "_generated", vocab_file_name
            )
        else:
            vocab_file_path = join(getcwd(), vocab_file_name)
        if not (exists(vocab_file_path)):
            self._export_vocabulary_file(vocab_file_path)

        self.__vocab_file_path = vocab_file_path
        return self.__vocab_file_path

    @source.setter
    def source(self, new_source):
        try:
            source_changed = self.__source != new_source
        except:
            source_changed = True

        if source_changed:
            self._reset(new_source)

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

    @gensim_model.setter
    def gensim_model(self, gensim_model):
        if not (self._source_is_path()):
            self.__gensim_model = gensim_model
        else:
            self.__gensim_model = None

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

    def _export_vocabulary_file(self, vocab_file_path):
        print(
            "Exporting {0} words to vocab file ({1})...".format(
                self.vocab_size, vocab_file_path
            )
        )
        start = time.time()
        cnt = 0
        makedirs(dirname(vocab_file_path), exist_ok=True)
        with open(vocab_file_path, "w") as f:
            for word in [*self.dictionary]:
                # if word != "<OOV>":
                if word != "<PAD>":
                    f.write("{0}\n".format(word))
                    cnt += 1
        time_taken = time.time() - start
        print("Exported {0} words in {1:.2f}sec".format(cnt, time_taken))

    def _reset(self, new_source):
        self.__source = new_source
        self.dictionary = None
        self.gensim_model = None

    def _load_embedding(self, path):
        dictionary = {}
        print("Loading embedding from file ({0})...".format(self.source))
        start = time.time()
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                values = line.strip().split()
                word = values[0]
                embedding = np.asarray(values[1:], dtype="float32")
                dictionary[word] = embedding
        time_taken = time.time() - start
        print(
            "Loaded {0} words in {1:.2f}sec".format(
                len(dictionary), time_taken
            )
        )

        return dictionary

    def _get_dict_from_gensim_model(self, source):
        emb_model = gensim_data.load(source)
        dictionary = {}
        for (word, vector) in zip([*emb_model.vocab], emb_model.vectors):
            dictionary[word] = vector

        return dictionary

    def _get_flags(self, dim_size):
        flags = {
            "<PAD>": np.zeros(shape=dim_size),
            "<OOV>": self.oov(dim_size=dim_size),
        }
        return flags

    def _source_is_path(self):
        return exists(self.source)
