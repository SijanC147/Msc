import os
import csv
import pickle
import spacy
import time
from utils import keep_token
from spacy.tokens import Doc
from spacy.attrs import ORTH # pylint: disable=E0611
from abc import ABC, abstractmethod

class Dataset(ABC):

    def __init__(self, train_file_path, eval_file_path, parent_folder='', debug_file_path="", embedding=None, rebuild_corpus=False):
        if len(parent_folder)==0:
            parent_folder = self.__class__.__name__
        else:
            parent_folder = os.path.join(self.__class__.__name__, parent_folder)
        self.parent_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', parent_folder)
        self.train_file_path = os.path.join(self.parent_directory, train_file_path)
        self.eval_file_path = os.path.join(self.parent_directory, eval_file_path)
        self.debug_file_path = os.path.join(self.parent_directory, debug_file_path)
        self.generated_data_directory = os.path.join(self.parent_directory, 'generated')
        self.corpus_file_path = os.path.join(self.parent_directory, self.generated_data_directory, 'corpus.csv')
        self.vocabulary_corpus = self.get_vocabulary_corpus(rebuild=rebuild_corpus)
        if embedding!=None:
            self.set_embedding(embedding)

    @abstractmethod
    def generate_dataset_dictionary(self, mode='debug'):
        pass

    def get_all_text_in_dataset(self):
        return set(self.get_dataset_dictionary(mode='train')['sentences']+self.get_dataset_dictionary(mode='eval')['sentences'])

    def get_mapped_features_and_labels(self, mode='debug'):
        self.load_embedding_from_corpus(self.vocabulary_corpus)

        if self.features_labels_save_file_exists(mode):
            features, labels = self.load_features_and_labels_from_save(mode)
        else:
            features = {
                'sentence' : [],
                'sentence_length': [],
                'target' : [],
                'mappings': {
                    'left': [],
                    'target': [],
                    'right': []
                },
                'maptest': []
            }
            labels = []

            token_to_ids_dict = self.get_word_to_id_dict(self.vocabulary_corpus, debug=(mode=='debug'))
            dataset_dict = self.get_dataset_dictionary(mode=mode)
            total_time = 0
            for index in range(len(dataset_dict['sentences'])):

                start = time.time()

                sentence = dataset_dict['sentences'][index].strip()
                target = dataset_dict['targets'][index].strip()
                label = int(dataset_dict['labels'][index].strip()) if type(dataset_dict['labels'][index])==str else dataset_dict['labels'][index]

                features['sentence'].append(sentence)
                features['sentence_length'].append(len(sentence))
                features['target'].append(target)
                labels.append(label)
                left_context, right_context = self.get_left_and_right_contexts(sentence=sentence, target=target, offset=dataset_dict.get('offset')) 
                features['mappings']['left'].append(self.embedding.map_embedding_ids(left_context.strip(), token_to_ids_dict=token_to_ids_dict))
                features['mappings']['right'].append(self.embedding.map_embedding_ids(right_context.strip(), token_to_ids_dict=token_to_ids_dict))
                features['mappings']['target'].append(self.embedding.map_embedding_ids(target, token_to_ids_dict=token_to_ids_dict))

                total_time += (time.time()-start)
                if(index%60==0):
                    print("Processed {0}/{1} lines ({2:.2f}%) tot:{3:.3f}s avg:{4:.3f}s/line".format(index+1, len(dataset_dict['sentences']), ((index+1)/len(dataset_dict['sentences']))*100, total_time, total_time/(index+1)))

            self.save_features_and_labels(mode, features, labels)            

        return features, labels

    def get_left_and_right_contexts(self, sentence, target, offset=None):
        if offset==None:
            left, _, right = sentence.partition(target) 
        else:
            left = sentence[:offset].strip()
            right = sentence[offset+len(target.strip()):].strip()
        return left, right

    def get_file(self, mode='debug'):
        if mode=='train':
            return self.train_file_path
        elif mode=='eval':
            return self.eval_file_path
        else:
            return self.debug_file_path

    def initialize_with_embedding(self, embedding):
        self.set_embedding(embedding)

    def set_embedding(self, embedding):
        self.embedding = embedding
        self.generated_embedding_directory = os.path.join(self.generated_data_directory, type(self.embedding).__name__, self.embedding.get_alias())
        self.embedding_id_file_path = os.path.join(self.generated_embedding_directory, 'words_to_ids.csv')
        self.projection_labels_file_path = os.path.join(self.generated_embedding_directory, 'tensorboard_projection_labels.tsv')
        self.partial_embedding_file_path = os.path.join(self.generated_embedding_directory, 'partial_'+self.embedding.get_version()+'.txt')
    
    def get_save_file_path(self, mode):
        return os.path.join(self.generated_embedding_directory, 'features_labels_'+mode+'.pkl')

    def get_dataset_dictionary_file_path(self, mode='debug'):
        return os.path.join(self.generated_data_directory, 'raw_dataset_dictionary_'+mode+'.pkl')

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
    
    def dataset_dictionary_file_exists(self, mode='debug'):
        return os.path.exists(self.get_dataset_dictionary_file_path(mode))

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

    def save_dataset_dictionary(self, dataset_dictionary, mode):
        os.makedirs(self.generated_data_directory, exist_ok=True)
        with open(self.get_dataset_dictionary_file_path(mode), 'wb+') as f:
            pickle.dump(dataset_dictionary, f, pickle.HIGHEST_PROTOCOL)
        
    def load_vocabulary_corpus_from_csv(self):
        vocabulary_corpus = {}
        if self.corpus_file_exists():
            with open(self.corpus_file_path) as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    vocabulary_corpus[row['word']]=int(row['count'])
        return vocabulary_corpus

    def load_dataset_dictionary_from_save(self, mode='debug'):
        if self.dataset_dictionary_file_exists(mode):
            with open(self.get_dataset_dictionary_file_path(mode), 'rb') as f:
                return pickle.load(f)
    
    def generate_vocabulary_corpus(self, source_documents):
        vocabulary_corpus = {}

        nlp = spacy.load('en')
        tokens = nlp(' '.join(map(lambda document: document.strip(), source_documents)))
        filtered_tokens = list(filter(keep_token, tokens))
        filtered_doc = nlp(' '.join(map(lambda token: token.text, filtered_tokens)))
        counts = filtered_doc.count_by(ORTH)
        os.makedirs(self.generated_data_directory, exist_ok=True)
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

    def get_word_to_id_dict(self, vocabulary_corpus, debug=False):
        token_to_ids_dict = {}
        if self.embedding_id_file_exists() and not(debug):
            with open(self.embedding_id_file_path) as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    token_to_ids_dict[row['word']] = row['id']
        else:
            print('Building token to ids dict...')
            start = time.time()
            token_to_ids_dict = self.embedding.map_token_ids_dict(vocabulary_corpus)
            print('Built token to ids dict in: ' + str(time.time()-start))
            if not(debug):
                os.makedirs(self.generated_embedding_directory, exist_ok=True)
                with open(self.embedding_id_file_path, 'w+') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=['word', 'id'])

                    writer.writeheader()
                    for word in [*token_to_ids_dict]:
                        writer.writerow({'word': word, 'id': token_to_ids_dict[word]})
        return token_to_ids_dict

    def get_vocabulary_corpus(self, rebuild=False):
        if self.corpus_file_exists() and not(rebuild):
            return self.load_vocabulary_corpus_from_csv()
        else:
            return self.generate_vocabulary_corpus(self.get_all_text_in_dataset())

    def get_dataset_dictionary(self, mode='debug'):
        if self.dataset_dictionary_file_exists(mode):
            return self.load_dataset_dictionary_from_save(mode)
        else:
            dataset_dictionary = self.generate_dataset_dictionary(mode)
            self.save_dataset_dictionary(dataset_dictionary=dataset_dictionary, mode=mode)
            return dataset_dictionary

    def load_embedding_from_corpus(self, vocabulary_corpus, force_rebuild_partial = False, force_rebuild_projection = False):
        if self.partial_embedding_file_exists() and not(force_rebuild_partial):
            partial_embedding_dict = self.embedding.load_embeddings_from_path(path=self.partial_embedding_file_path)
        else:
            partial_embedding_dict = self.embedding.filter_on_corpus([*vocabulary_corpus])
            self.generate_partial_embedding_file(partial_embedding_dict)
        
        if not(self.projection_labels_file_exists()) or force_rebuild_projection:
            self.generate_projection_labels_file(partial_embedding_dict)