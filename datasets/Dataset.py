import os
import csv
import pickle
import spacy
import time
from spacy.tokens import Doc
from spacy.attrs import ORTH
from abc import ABC, abstractmethod

class Dataset(ABC):

    def __init__(self, embedding, parent_folder, train_file_path, eval_file_path, debug_file_path=""):
        self.embedding = embedding
        self.parent_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', parent_folder)
        self.train_file_path = os.path.join(self.parent_directory, train_file_path)
        self.eval_file_path = os.path.join(self.parent_directory, eval_file_path)
        self.debug_file_path = os.path.join(self.parent_directory, debug_file_path)
        self.generated_data_directory = os.path.join(self.parent_directory, 'generated')
        self.corpus_file_path = os.path.join(self.parent_directory, self.generated_data_directory, 'corpus.csv')
        self.generated_embedding_directory = os.path.join(self.generated_data_directory, type(self.embedding).__name__, self.embedding.get_alias())
        self.embedding_id_file_path = os.path.join(self.generated_embedding_directory, 'words_to_ids.csv')
        self.projection_labels_file_path = os.path.join(self.generated_embedding_directory, 'tensorboard_projection_labels.tsv')
        self.partial_embedding_file_path = os.path.join(self.generated_embedding_directory, 'partial_'+self.embedding.get_version()+'.txt')

    @abstractmethod
    def get_vocabulary_corpus(self, rebuild=False):
        pass

    @abstractmethod
    def get_mapped_features_and_labels(self, mode='debug'):
        pass

    def get_file(self, mode='debug'):
        if mode=='train':
            return self.train_file_path
        elif mode=='eval':
            return self.eval_file_path
        else:
            return self.debug_file_path
    
    def get_save_file_path(self, mode):
        return os.path.join(self.generated_embedding_directory, 'features_labels_'+mode+'.pkl')

    def corpus_file_exists(self):
        return os.path.isfile(self.corpus_file_path)

    def embedding_id_file_exists(self):
        return os.path.exists(self.embedding_id_file_path)
    
    def partial_embedding_file_exists(self):
        return os.path.exists(self.partial_embedding_file_path)
    
    def projection_labels_file_exists(self):
        return os.path.exists(self.projection_labels_file_path)

    def features_labels_save_file_exists(self, mode):
        return os.path.exists(self.get_save_file_path(mode))
    
    def load_features_and_labels_from_save(self, mode):
        if self.features_labels_save_file_exists(mode):
            with open(self.get_save_file_path(mode), 'rb') as f:
                saved_data = pickle.load(f)
                features = saved_data['features']
                labels = saved_data['labels']
                return features, labels

    def save_features_and_labels(self, mode, features, labels):
            saved_data = {
                'features': features,
                'labels': labels
            }

            with open(self.get_save_file_path(mode), 'wb+') as f:
                pickle.dump(saved_data, f, pickle.HIGHEST_PROTOCOL)

    def load_vocabulary_corpus_from_csv(self):
        vocabulary_corpus = {}
        if self.corpus_file_exists():
            with open(self.corpus_file_path) as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    vocabulary_corpus[row['word']]=int(row['count'])
        return vocabulary_corpus

    def generate_vocabulary_corpus(self, source_documents):
        vocabulary_corpus = {}

        nlp = spacy.load('en')
        doc = nlp(' '.join(source_documents))
        counts = doc.count_by(ORTH)
        with open(self.corpus_file_path, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=['word', 'count'])

            writer.writeheader()
            for word_id, count in sorted(counts.items(), reverse=True, key=lambda item: item[1]):
                writer.writerow({'word': nlp.vocab.strings[word_id], 'count': count})
                vocabulary_corpus[nlp.vocab.strings[word_id]]=count
        
        return vocabulary_corpus

    def generate_partial_embedding_file(self, partial_embedding):
        os.makedirs(self.generated_embedding_directory, exist_ok=True)
        with open(self.partial_embedding_file_path, 'w+') as f:
            for word in [*partial_embedding]:
                f.write(word + ' ' + ' '.join(partial_embedding[word].astype(str)) + '\n')

    def generate_projection_labels_file(self, partial_embedding):
        os.makedirs(self.generated_embedding_directory, exist_ok=True)
        with open(self.projection_labels_file_path, 'w+') as f:
            f.write('Words')
            for word in [*partial_embedding]:
                f.write(word + '\n')

    def get_word_to_id_dict(self, vocabulary_corpus):
        token_to_ids_dict = {}
        if self.embedding_id_file_exists():
            with open(self.embedding_id_file_path) as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    token_to_ids_dict[row['word']] = row['id']
        else:
            print('Building token to ids dict...')
            start = time.time()
            token_to_ids_dict = self.embedding.map_token_ids_dict(vocabulary_corpus)
            print('Built token to ids dict in: ' + str(time.time()-start))
            os.makedirs(self.generated_embedding_directory, exist_ok=True)
            with open(self.embedding_id_file_path, 'w+') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=['word', 'id'])

                writer.writeheader()
                for word in [*token_to_ids_dict]:
                    writer.writerow({'word': word, 'id': token_to_ids_dict[word]})
        return token_to_ids_dict

    def load_embedding_from_corpus(self, vocabulary_corpus, force_rebuild_partial = False, force_rebuild_projection = False):
        if self.partial_embedding_file_exists() and not(force_rebuild_partial):
            partial_embedding_dict = self.embedding.load_embeddings_from_path(path=self.partial_embedding_file_path)
        else:
            partial_embedding_dict = self.embedding.filter_on_corpus([*vocabulary_corpus])
            self.generate_partial_embedding_file(partial_embedding_dict)
        
        if not(self.projection_labels_file_exists()) or force_rebuild_projection:
            self.generate_projection_labels_file(partial_embedding_dict)