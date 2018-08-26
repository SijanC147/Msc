import re
import numpy as np
from utils import tokenize_phrase

class Embedding:

    def __init__(self, embedding_path):
        embeddings = {}

        with open('embeddings/data/'+embedding_path, "r", encoding='utf-8') as f:
            for line in f:
                values = line.strip().split()
                word = values[0]
                embedding = np.asarray(values[1:], dtype='float32')
                embeddings[word] = embedding

        self.embedding_dict = embeddings

    def get_embedding_dictionary(self):
        return self.embedding_dict

    def get_embedding_vectors(self):
        return np.asarray(list(self.embedding_dict.values()))

    def get_embedding_vocab(self):
        return np.asarray([*self.embedding_dict])

    def map_embedding_ids(self, phrase, separate_on=""):
        phrase = str(phrase, 'utf-8') if type(phrase)!=str else phrase
        if len(separate_on)==0:
            return list([*self.embedding_dict].index(w) for w in tokenize_phrase(phrase.lower()) if w in [*self.embedding_dict])
        else:
            left_mapped_ids = list([*self.embedding_dict].index(w) for w in tokenize_phrase(re.match(r"(.*)"+re.escape(separate_on)+"(.*)",phrase).group(1).lower()) if w in [*self.embedding_dict])
            right_mapped_ids = list([*self.embedding_dict].index(w) for w in tokenize_phrase(re.match(r"(.*)"+re.escape(separate_on)+"(.*)",phrase).group(2).lower()) if w in [*self.embedding_dict])
            return left_mapped_ids, right_mapped_ids