import re

import numpy as np
import tensorflow as tf

from utils import tokenize_phrase


class Embedding:

    def __init__(self, path, alias, version):
        self.alias = alias
        self.version = version
        self.path = path
        self.embedding_dict = {}

    def load_embeddings_from_path(self, path='default'):
        if path!='default':
            load_path = path 
        else:
            load_path = 'embeddings/data/'+self.path

        with open(load_path, "r", encoding='utf-8') as f:
            for line in f:
                values = line.strip().split()
                word = values[0]
                embedding = np.asarray(values[1:], dtype='float32')
                self.embedding_dict[word] = embedding
        
        return self.embedding_dict

    def get_alias(self):
        return self.alias

    def get_version(self):
        return self.version

    def get_embedding_dictionary(self):
        return self.embedding_dict

    def get_embedding_vectors(self):
        return np.asarray(list(self.embedding_dict.values()))

    def get_embedding_vocab(self):
        return np.asarray([*self.embedding_dict])

    def get_vocab_size(self):
        return len(self.embedding_dict)

    def get_embedding_dim(self):
        return int((self.get_embedding_vectors().size)/self.get_vocab_size())

    def set_embedding_matrix_variable(self):
        with tf.variable_scope('shared', reuse=tf.AUTO_REUSE):
            embedding_matrix = tf.get_variable(name='embedding_matrix', shape=[self.get_vocab_size(),self.get_embedding_dim()], initializer=tf.constant_initializer(self.get_embedding_vectors()), trainable=False)
        
        return embedding_matrix

    def map_embedding_ids(self, phrase, token_to_ids_dict={}):
        phrase = str(phrase, 'utf-8') if type(phrase)!=str else phrase

        if len(token_to_ids_dict)==0:
            return list([*self.embedding_dict].index(token) for token in tokenize_phrase(phrase.lower()) if token in [*self.embedding_dict])
        else:
            return list(token_to_ids_dict[token] for token in tokenize_phrase(phrase.lower()) if token in [*token_to_ids_dict])

    def map_token_ids_dict(self, tokens):
        return {token: [*self.embedding_dict].index(token) for token in tokens if token in [*self.embedding_dict]}

    def map_token_ids_list(self, tokens):
        return list([*self.embedding_dict].index(token) for token in tokens if token in [*self.embedding_dict])

    def map_token_embeddings(self, tokens):
        return list(self.embedding_dict[token] for token in tokens if token in [*self.embedding_dict])

    def filter_on_corpus(self, tokens):
        self.load_embeddings_from_path()
        self.embedding_dict = {token: self.embedding_dict[token] for token in tokens if token in [*self.embedding_dict]}
        return self.embedding_dict
