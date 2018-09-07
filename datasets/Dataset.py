import os
import csv
import pickle
import spacy
import time
import math
import random
from statistics import mean
from utils import keep_token
from spacy.tokens import Doc
from spacy.attrs import ORTH # pylint: disable=E0611
from abc import ABC, abstractmethod

class Dataset(ABC):
    def __init__(self, train_file_path, eval_file_path, parent_folder='', debug_file_path="", embedding=None, rebuild_corpus=False):
        """Creates a new dataset based on the specified training and evaluation data files
        
        Arguments:
            ABC {Class} -- makes the class abstract
            train_file_path {str} -- file path for the data to use when mode=='train'
            eval_file_path {str} -- file path for the data to use when mode=='eval'
        
        Keyword Arguments:
            parent_folder {str} --  specify further subfolders if the dataset is further stratified (default: {''})
            debug_file_path {str} -- specify any particular data file used for debugging only (default: {""})
            embedding {Embedding} -- optional embedding to attach at initialization, can be attached later (default: {None})
            rebuild_corpus {bool} -- flag to rebuild corpus csv file even if one exists already (default: {False})
        """
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
    def generate_dataset_dictionary(self, mode):
        """Loads data from raw data files into a dataset dictionary
        
            This must be implemented for every new dataset. 
            All other methods and functionality build on top of this function

            When adding a new dataset, you only need to defined how to load the data into the dataset dictionary 
            in this function.

            For each new dataset, this function must return a dataset_dictionary in the form of
            {
                'sentences':[list of str],
                'targets':[list of str],
                'labels':[list of {-1,0,1} for positive, neutral and negative respectively]
                (optionally)'offsets':[list of int, marking the position where a target starts in the sentence]
            }

            For sentences with multiple targets, there must be a separate entry in the dictionary for 
            each (sentence,target|offset,label) trio.

        Arguments:
            mode {'train','eval','debug'} -- Which raw data file to load from depends on this
        """
        pass

    def get_all_text_in_dataset(self):
        """Returns list of all unique sentences in dataset.
        
        Returns:
            list -- Unique sentences in dataset
        """
        return set(self.get_dataset_dictionary(mode='train')['sentences']+self.get_dataset_dictionary(mode='eval')['sentences'])

    def get_mapped_features_and_labels(self, mode):
        """Gets features and labels in form of IDs from word embeddin
        
        Arguments:
            mode {'train','eval','debug'} -- controls features and labels source
        
        Returns:
            (dict, list) -- feature dictionary and labels list
        """
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
            }
            labels = []

            word_to_ids_dict = self.get_word_to_id_dict(self.vocabulary_corpus, debug=(mode=='debug'))
            dataset_dict = self.get_dataset_dictionary(mode=mode)
            total_time = 0
            for index in range(len(dataset_dict['sentences'])):

                start = time.time()

                sentence = dataset_dict['sentences'][index].strip()
                target = dataset_dict['targets'][index].strip()
                label = int(dataset_dict['labels'][index].strip()) if type(dataset_dict['labels'][index])==str else dataset_dict['labels'][index]

                features['sentence'].append(sentence)
                features['target'].append(target)
                labels.append(label)
                left_context, right_context = self.get_left_and_right_contexts(sentence=sentence, target=target, offset=dataset_dict.get('offset')) 

                left_mapping = self.embedding.map_embedding_ids(left_context.strip(), word_to_ids_dict=word_to_ids_dict)
                target_mapping = self.embedding.map_embedding_ids(target, word_to_ids_dict=word_to_ids_dict)
                right_mapping = self.embedding.map_embedding_ids(right_context.strip(), word_to_ids_dict=word_to_ids_dict)

                features['sentence_length'].append(len(left_mapping+target_mapping+right_mapping))

                features['mappings']['left'].append(left_mapping)
                features['mappings']['target'].append(target_mapping)
                features['mappings']['right'].append(right_mapping)

                total_time += (time.time()-start)
                if(index%60==0):
                    print("Processed {0}/{1} lines ({2:.2f}%) tot:{3:.3f}s avg:{4:.3f}s/line".format(index+1, len(dataset_dict['sentences']), ((index+1)/len(dataset_dict['sentences']))*100, total_time, total_time/(index+1)))

            self.save_features_and_labels(mode, features, labels)            

        return features, labels

    def get_left_and_right_contexts(self, sentence, target, offset=None):
        """Gets contexts of target in sentence
        
        Arguments:
            sentence {str} -- Sentence containing target
            target {str} -- Target in the sentence
        
        Keyword Arguments:
            offset {int} -- Target location instead of the target phrase (default: {None})
        
        Returns:
            (str, str) -- left and right contexts
        """
        if offset==None:
            left, _, right = sentence.partition(target) 
        else:
            left = sentence[:offset].strip()
            right = sentence[offset+len(target.strip()):].strip()
        return left, right

    def get_file(self, mode):
        """Retrieves the respective data file for the dataset
        
        Arguments:
            mode {'train','eval','debug'} -- Defines which source file to retrieve 
        
        Returns:
            str -- path to souce file
        """
        if mode=='train':
            return self.train_file_path
        elif mode=='eval':
            return self.eval_file_path
        else:
            return self.debug_file_path

    def initialize_with_embedding(self, embedding):
        """Initializes embedding settinns for the dataset
        
        Arguments:
            embedding {Embedding} -- the embedding to attach  
        """
        self.set_embedding(embedding)

    def set_embedding(self, embedding):
        """Sets up the necessary settings based on a provided Embedding
        
        Arguments:
            embedding {Embedding} -- the embedding being attached to the dataset
        """
        self.embedding = embedding
        self.generated_embedding_directory = os.path.join(self.generated_data_directory, type(self.embedding).__name__, self.embedding.get_alias())
        self.embedding_id_file_path = os.path.join(self.generated_embedding_directory, 'words_to_ids.csv')
        self.projection_labels_file_path = os.path.join(self.generated_embedding_directory, 'tensorboard_projection_labels.tsv')
        self.partial_embedding_file_path = os.path.join(self.generated_embedding_directory, 'partial_'+self.embedding.get_version()+'.txt')
        self.embedding.init_partial_embedding_if_exists(partial_embedding_path=self.partial_embedding_file_path)
    
    def get_save_file_path(self, mode):
        """Generates file name to save obtained features and labels for future re-use
        
        Arguments:
            mode {'train','eval','debug'} -- source of features and labels depends on this
        
        Returns:
            str -- absolute path to save file
        """
        return os.path.join(self.generated_embedding_directory, 'features_labels_'+mode+'.pkl')

    def get_dataset_dictionary_file_path(self, mode):
        """Get the file path for a saved copy of the data dictionary
        
        Arguments:
            mode {'train','eval','debug'} -- the source of the data dictionary would depend on this
        
        Returns:
            str -- absolute path to the data dictionary file
        """
        return os.path.join(self.generated_data_directory, 'raw_dataset_dictionary_'+mode+'.pkl')

    def corpus_file_exists(self):
        """Check if a corpus file has been generated for this dataset
        
        Returns:
            bool -- wether the corpus file exists
        """
        return os.path.isfile(self.corpus_file_path)

    def embedding_id_file_exists(self):
        """Check if a word to id (index in embedding) file exists
        
        Returns:
            bool -- wether the word to id file exists
        """
        return os.path.exists(self.embedding_id_file_path)
    
    def partial_embedding_file_exists(self):
        """Check if a partial embedding file has been generated before for reuse
        
        Returns:
            bool -- wether the partial embedding file exists
        """
        return os.path.exists(self.partial_embedding_file_path)
    
    def projection_labels_file_exists(self):
        """Checks if a projection file exists, to provide labels for tensorboard embedding projector
        
        Returns:
            bool -- wether the projection label file exists
        """
        return os.path.exists(self.projection_labels_file_path)

    def features_labels_save_file_exists(self, mode):
        """Checks if a saved version of the features and labels for a mode exists
        
        Arguments:
            mode {'train','eval','debug'} -- the source of the features and labels depend on this
        
        Returns:
            bool -- wether the saved features and labels file exists
        """
        return os.path.exists(self.get_save_file_path(mode))
    
    def dataset_dictionary_file_exists(self, mode):
        """Checks if a saved version of the dataset dictionary exists for a specified mode
        
        Arguments:
            mode {'train','eval','debug'} -- the source of the dataset dictionary depends on this 
        
        Returns:
            bool -- wether the dataset dictionary file exists
        """
        return os.path.exists(self.get_dataset_dictionary_file_path(mode))

    def load_features_and_labels_from_save(self, mode):
        """Loads features and labels for a specific mode from a previous save file
        
        Arguments:
            mode {'train','eval','debug'} -- the features and labels file to load depends on this
        
        Returns:
            (dict, list) -- loaded features dictionary and list of labels
        """
        if self.features_labels_save_file_exists(mode):
            with open(self.get_save_file_path(mode), 'rb') as f:
                saved_data = pickle.load(f)
                features = saved_data['features']
                labels = saved_data['labels']
                return features, labels

    def save_features_and_labels(self, mode, features, labels):
        """Saves features and labels for efficient re-use in future experiments
        
        Arguments:
            mode {'train','eval','debug'} -- appended to the save file name
            features {dict} -- features to save
            labels {list of {-1,0,1}} -- corresponding labels to save
        """
        saved_data = {
            'features': features,
            'labels': labels
        }

        with open(self.get_save_file_path(mode), 'wb+') as f:
            pickle.dump(saved_data, f, pickle.HIGHEST_PROTOCOL)

    def save_dataset_dictionary(self, dataset_dictionary, mode):
        """Saves a dataset dictionary to external file
        
        Arguments:
            dataset_dictionary {dict} -- contains sentences, targets, labels, and possibly offsets
            mode {'train','eval','debug'} -- appended to the save file name
        """
        os.makedirs(self.generated_data_directory, exist_ok=True)
        with open(self.get_dataset_dictionary_file_path(mode), 'wb+') as f:
            pickle.dump(dataset_dictionary, f, pickle.HIGHEST_PROTOCOL)
        
    def load_vocabulary_corpus_from_csv(self):
        """Loads a previously saved corpus of unique token in the dataset
        
        Returns:
            dict -- dictionary of all unique tokens in the dataset
        """
        vocabulary_corpus = {}
        if self.corpus_file_exists():
            with open(self.corpus_file_path) as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    vocabulary_corpus[row['word']]=int(row['count'])
        return vocabulary_corpus

    def load_dataset_dictionary_from_save(self, mode):
        """Loads a previously saved dataset dictionary from file
        
        Arguments:
            mode {'train','eval','debug'} -- specifies which file to load from
        
        Returns:
            dict -- dictionary of sentences, targets, labels and possibly offsets
        """
        if self.dataset_dictionary_file_exists(mode):
            with open(self.get_dataset_dictionary_file_path(mode), 'rb') as f:
                return pickle.load(f)
    
    def generate_vocabulary_corpus(self, source_documents):
        """Generates list of all unique tokens in the dataset
        
        Arguments:
            source_documents {list of str} -- all unique sentences in the dataset
        
        Returns:
            dict -- dictionary of all unique tokens in the dataset
        """
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
        """Saves a partial embedding file created from the original embedding and tokens in this dataset
        
        Arguments:
            partial_embedding {dict} -- a subset of the original embedding matrix
        """
        os.makedirs(self.generated_embedding_directory, exist_ok=True)
        with open(self.partial_embedding_file_path, 'w+') as f:
            for word in [*partial_embedding]:
                if word!='<OOV>' and word!='<PAD>':
                    f.write(word + ' ' + ' '.join(partial_embedding[word].astype(str)) + '\n')

    def generate_projection_labels_file(self, partial_embedding):
        """Generates the embedding projection labels for tensorboard based on the partial embedding
        
        Arguments:
            partial_embedding {dict} -- a subset of the original embedding
        """
        os.makedirs(self.generated_embedding_directory, exist_ok=True)
        with open(self.projection_labels_file_path, 'w+') as f:
            f.write('Words')
            for word in [*partial_embedding]:
                f.write(word + '\n')

    def get_word_to_id_dict(self, vocabulary_corpus, debug=False):
        """Gets a dictionary mapping words to indices in the partial embedding
        
        Arguments:
            vocabulary_corpus {dict} -- the corpus of unique tokens appearing in this dataset
        
        Keyword Arguments:
            debug {bool} -- Only used in special cases where a specific embedding with different indices (for dev) is used (default: {False})
        
        Returns:
            dict -- dictionary of unique tokens and their respective id (index) in the partial embedding
        """
        token_to_ids_dict = {}
        if self.embedding_id_file_exists() and not(debug):
            with open(self.embedding_id_file_path) as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    token_to_ids_dict[row['word']] = row['id']
        else:
            print('Building token to ids dict...')
            start = time.time()
            token_to_ids_dict = self.embedding.get_word_to_ids_dict(vocabulary_corpus)
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
        """Returns the vocabulary corpus from the save file, or rebuilds it from scratch
        
        Keyword Arguments:
            rebuild {bool} -- Wether to rebuild the vocabulary corpus even if a save file exists (default: {False})
        
        Returns:
            dict -- dictionary of all unique tokens in the dataset
        """
        if self.corpus_file_exists() and not(rebuild):
            return self.load_vocabulary_corpus_from_csv()
        else:
            return self.generate_vocabulary_corpus(self.get_all_text_in_dataset())

    def get_dataset_dictionary(self, mode):
        """Returns the specific data dictionary for a particular mode, from a save file, or generates it otherwise
        
        Arguments:
            mode {'train','eval','debug'} -- the specific mode for the dataset dicitonary, which would be in the file name
        
        Returns:
            dict -- dictionary of sentences, targets, labels and possibly offsets
        """
        if self.dataset_dictionary_file_exists(mode):
            return self.load_dataset_dictionary_from_save(mode)
        else:
            dataset_dictionary = self.generate_dataset_dictionary(mode)
            self.save_dataset_dictionary(dataset_dictionary=dataset_dictionary, mode=mode)
            return dataset_dictionary

    def load_embedding_from_corpus(self, vocabulary_corpus, force_rebuild_partial = False, force_rebuild_projection = False):
        """Returns the partial embedding if it exists in a save file, or builds it from scratch and saves it
        
        Arguments:
            vocabulary_corpus {dict} -- dictionary of unique tokens in the dataset
        
        Keyword Arguments:
            force_rebuild_partial {bool} -- wether to rebuild the partial embedding even if a save file exists (default: {False})
            force_rebuild_projection {bool} -- wether to rebuild the projection label tsv file even if it already exists (default: {False})
        """
        if self.partial_embedding_file_exists() and not(force_rebuild_partial):
            partial_embedding_dict = self.embedding.load_embeddings_from_path(path=self.partial_embedding_file_path)
        else:
            partial_embedding_dict = self.embedding.filter_on_corpus([*vocabulary_corpus])
            self.generate_partial_embedding_file(partial_embedding_dict)
        
        if not(self.projection_labels_file_exists()) or force_rebuild_projection:
            self.generate_projection_labels_file(partial_embedding_dict)