import re
import os

import numpy as np
import tensorflow as tf

from utils import tokenize_phrase


class Embedding:
    def __init__(self, path, alias, version, oov_embedding=None):
        self.alias = alias
        self.version = version
        self.default_path = path
        self.dictionary = {}
        if oov_embedding is None:
            self.oov_embedding = lambda embedding_dim: np.random.uniform(
                low=-1, high=1, size=embedding_dim
            )
        else:
            self.oov_embedding = oov_embedding

    def load_embeddings_from_path(self, path=None):
        if path is not None:
            load_path = path
        else:
            load_path = os.path.join("embeddings", "data", self.default_path)

        with open(load_path, "r", encoding="utf-8") as f:
            for line in f:
                values = line.strip().split()
                word = values[0]
                embedding = np.asarray(values[1:], dtype="float32")
                self.dictionary[word] = embedding

        self.dimension_size = len(next(iter(self.dictionary.values())))
        self.dictionary = {
            "<PAD>": np.zeros(shape=self.dimension_size),
            "<OOV>": self.oov_embedding(embedding_dim=self.dimension_size),
            **self.dictionary,
        }
        self.vocab_size = len(self.dictionary)

        return self.dictionary

    def init_partial_embedding_if_exists(self, partial_embedding_path):
        if os.path.exists(partial_embedding_path):
            self.load_embeddings_from_path(path=partial_embedding_path)

    def get_tf_embedding_initializer(self):
        def embedding_initializer(
            shape=(self.vocab_size, self.dimension_size),
            dtype=tf.float32,
            partition_info=None,
        ):
            return np.asarray(
                [self.dictionary[word] for word in [*self.dictionary]]
            )

        return embedding_initializer

    def map_embedding_ids(self, phrase, word_to_ids_dict={}):
        phrase = str(phrase, "utf-8") if type(phrase) != str else phrase

        if len(word_to_ids_dict) == 0:
            return list(
                [*self.dictionary].index(word)
                if word in [*self.dictionary]
                else [*self.dictionary].index("<OOV>")
                for word in tokenize_phrase(phrase.lower())
            )
        else:
            return list(
                word_to_ids_dict[word]
                for word in tokenize_phrase(phrase.lower())
                if word in [*word_to_ids_dict]
            )

    def get_word_to_ids_dict(self, words):
        return {
            word: [*self.dictionary].index(word)
            if word in [*self.dictionary]
            else [*self.dictionary].index("<OOV>")
            for word in words
        }

    def filter_on_corpus(self, tokens):
        self.load_embeddings_from_path()
        self.dictionary = {
            token: self.dictionary[token]
            for token in tokens
            if token in [*self.dictionary]
        }
        self.dictionary = {
            "<PAD>": np.zeros(shape=self.dimension_size),
            "<OOV>": self.oov_embedding(embedding_dim=self.dimension_size),
            **self.dictionary,
        }
        self.vocab_size = len(self.dictionary)
        return self.dictionary
