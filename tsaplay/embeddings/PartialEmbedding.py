import numpy as np
import tensorflow as tf
import time
import gensim.downloader as gensim_data
from os import makedirs, getcwd
from os.path import join, normpath, basename, splitext, dirname, exists
from tsaplay.utils._nlp import tokenize_phrase, default_oov
from tsaplay.utils._io import gprint, write_embedding_to_disk
from tsaplay.embeddings.Embedding import Embedding, EMBEDDINGS


class PartialEmbedding(Embedding):
    def __init__(self, partial_path, vocab_list, source, oov=None):
        self.__partial_path = partial_path
        self.__vocab_list = vocab_list
        super().__init__(source, oov)

    @property
    def partial_path(self):
        return self.__partial_path

    @property
    def name(self):
        return "partial_" + self.__source

    @property
    def data_dir(self):
        return join(self.__partial_path, self.source)

    @property
    def dim_size(self):
        return len(next(iter(self.dictionary.values())))

    @property
    def vectors(self):
        return np.asarray([self.dictionary[w] for w in [*self.dictionary]])

    @source.setter
    def source(self, new_source):
        try:
            self.__source = new_source
            partial_save_file = join(self.data_dir, "_partial.txt")
            if exists(partial_save_file):
                self.__dictionary = {
                    **self._get_flags(self.dim_size),
                    **self._load_dictionary_from_file(partial_save_file),
                }
            else:
                self.__gensim_model = gensim_data.load(self.__source)
                dictionary = self._build_embedding_dictionary()
                self.__dictionary = {
                    **self._get_flags(self.dim_size),
                    **self._filter_on_vocab(dictionary),
                }
                self._write_embedding_to_file()
                self._export_vocabulary_files()
        except:
            raise ValueError("Invalid source {0}".format(new_source))

    def _filter_on_vocab(self, dictionary):
        vocab = [word.lower() for word in self.__vocab_list]
        return {
            word: dictionary[word] for word in vocab if word in [*dictionary]
        }

    def _load_dictionary_from_file(self, path):
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

    def _write_embedding_to_file(self):
        partial_save_file = join(self.data_dir, "_partial.txt")
        with open(partial_save_file, "w+") as f:
            for word in [*self.__dictionary]:
                if word != "<OOV>" and word != "<PAD>":
                    vector = " ".join(self.__dictionary[word].astype(str))
                    f.write("{w} {v}\n".format(w=word, v=vector))
