import numpy as np

class Embedding:

    def __init__(self, embedding_path):
        embeddings = {}

        with open('embeddings/'+embedding_path, "r", encoding='utf-8') as f:
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

    def map_embedding_ids(self, phrases):
        if type(phrases) is str:
            phrases = [phrases]

        all_mapped_ids = list()
        for phrase in phrases:
            phrase_mapped_ids = list([*self.embedding_dict].index(w) for w in phrase.lower().split() if w in [*self.embedding_dict])
            all_mapped_ids.append(phrase_mapped_ids)
        
        if type(phrases) is str:
            return all_mapped_ids[0]
        else:
            return all_mapped_ids
