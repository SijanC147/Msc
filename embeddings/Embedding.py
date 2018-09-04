import re

import numpy as np
import tensorflow as tf

from utils import tokenize_phrase


class Embedding:

    def __init__(self, path, alias, version):
        """Create a new embedding
        
        Arguments:
            path {str} -- path to the particular embedding data file
            alias {str} -- a specific alias used as a shorthand to identify the embedding
            version {str} -- an additional shorthand used to further delinieate between different embedding versions
        """
        self.alias = alias
        self.version = version
        self.path = path
        self.embedding_dict = {}

    def load_embeddings_from_path(self, path='default'):
        """Populate the embedding dictionary from the external data file
        
        Keyword Arguments:
            path {str} -- Specify a different file to the default from where to load the embedding dictionary, such as a saved partial embedding (default: {'default'})
        
        Returns:
            dict -- The embedding matrix in dictionary {word:embedding} format
        """
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
        """Returns the alias value for the embedding.
        
        Returns:
            str -- The alias for the embeddin
        """ 
        return self.alias

    def get_version(self):
        """Returns the version for the embedding.
        
        Returns:
            str -- The version for the embedding
        """ 
        return self.version

    def get_embedding_dictionary(self):
        """Returns the embedding in dictionary format
        
        Returns:
            dict -- dictionary of word:embedding 
        """
        return self.embedding_dict

    def get_embedding_vectors(self):
        """Returns only the embedding vectors
        
        Returns:
            numpy.array -- numpy array of vector values
        """
        return np.asarray(list(self.embedding_dict.values()))

    def get_embedding_vocab(self):
        """Returns only the words in the embedding
        
        Returns:
            numpy.array -- array of the words (keys) of the embedding dictionary
        """
        return np.asarray([*self.embedding_dict])

    def get_vocab_size(self):
        """Returns the size of the embedding
        
        Returns:
            int -- number of entries in the embedding matrix
        """
        return len(self.embedding_dict)

    def get_embedding_dim(self):
        """Returns the embedding dimension of the embedding matrix
        
        Returns:
            int -- embedding dimension of the embedding matrix
        """
        return int((self.get_embedding_vectors().size)/self.get_vocab_size())

    def set_embedding_matrix_variable(self):
        """Initializes the embedding matrix on the TensorFlow graph in a SHARED scope
        
        Returns:
            tf.variable -- tensorflow variable of the embedding matrix
        """
        with tf.variable_scope('shared', reuse=tf.AUTO_REUSE):
            embedding_matrix = tf.get_variable(name='embedding_matrix', shape=[self.get_vocab_size(),self.get_embedding_dim()], initializer=tf.constant_initializer(self.get_embedding_vectors()), trainable=False)
        
        return embedding_matrix

    def map_embedding_ids(self, phrase, token_to_ids_dict={}):
        """Maps a phrase of tokens to IDs (indices) in the embedding matrix
        
        Arguments:
            phrase {str} -- string of tokens to map to IDs
        
        Keyword Arguments:
            token_to_ids_dict {dict} -- a pre-loaded token to ID dictionary from previous runs for efficiency (default: {{}})
        
        Returns:
            list of int -- the IDs (indices) of each token in the phrase in the embedding matrix
        """
        phrase = str(phrase, 'utf-8') if type(phrase)!=str else phrase

        if len(token_to_ids_dict)==0:
            return list([*self.embedding_dict].index(token) for token in tokenize_phrase(phrase.lower()) if token in [*self.embedding_dict])
        else:
            return list(token_to_ids_dict[token] for token in tokenize_phrase(phrase.lower()) if token in [*token_to_ids_dict])

    def map_token_ids_dict(self, tokens):
        """Get dictionary of token IDs indexed by the token text
        
        Arguments:
            tokens {list} -- list of tokens to map to ID in the embedding matrix
        
        Returns:
            dict -- dictionary of IDs indexed by the token
        """
        return {token: [*self.embedding_dict].index(token) for token in tokens if token in [*self.embedding_dict]}

    def map_token_ids_list(self, tokens):
        """Get list of IDs (indices) of each token in the embedding matrix
        
        Arguments:
            tokens {list} -- tokens to map to IDs 
        
        Returns:
            list -- list of IDs of the provided tokens
        """
        return list([*self.embedding_dict].index(token) for token in tokens if token in [*self.embedding_dict])

    def map_token_embeddings(self, tokens):
        """Get list embedding vectors for tokens
        
        Arguments:
            tokens {list} -- tokens to embed
        
        Returns:
            list -- embeddings for tokens based on the mebedding matrix
        """
        return list(self.embedding_dict[token] for token in tokens if token in [*self.embedding_dict])

    def filter_on_corpus(self, tokens):
        """Filter a large embedding on a subset of tokens to speed up lookup performance
        
        Arguments:
            tokens {list} -- corpus to subset full embedding dictionary on
        
        Returns:
            dict -- dictionary of the partial embedding {word:embedding} based on the provided tokens
        """
        self.load_embeddings_from_path()
        self.embedding_dict = {token: self.embedding_dict[token] for token in tokens if token in [*self.embedding_dict]}
        return self.embedding_dict
