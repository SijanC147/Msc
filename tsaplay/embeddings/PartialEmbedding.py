import numpy as np
import gensim.downloader as gensim_data
from os.path import join, exists
from tsaplay.embeddings.Embedding import Embedding, EMBEDDINGS


class PartialEmbedding(Embedding):
    def __init__(self, partial_name, vocab_list, source, oov=None):
        self.__partial_name = partial_name.lower()
        self.__vocab_list = vocab_list
        super().__init__(source, oov)

    @property
    def name(self):
        return "_".join([self.__partial_name, self._source])

    def _filter_on_vocab(self, dictionary):
        vocab = [word.lower() for word in self.__vocab_list]
        return {
            word: dictionary[word] for word in vocab if word in [*dictionary]
        }

    def _load_dictionary_from_file(self, path):
        dictionary = {}
        print("Loading embedding from file ({0})...".format(path))
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                values = line.strip().split()
                word = values[0]
                embedding = np.asarray(values[1:], dtype="float32")
                dictionary[word] = embedding

        return dictionary

    # def _write_embedding_to_file(self):
    #     partial_save_file = join(self.data_dir, "_partial.txt")
    #     with open(partial_save_file, "w+") as f:
    #         for word in [*self._dictionary]:
    #             if word != "<OOV>" and word != "<PAD>":
    #                 vector = " ".join(self._dictionary[word].astype(str))
    #                 f.write("{w} {v}\n".format(w=word, v=vector))
