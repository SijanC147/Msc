import fileinput
import spacy
import re
import csv
import time
import pickle
import os.path
from datasets.Dataset import Dataset
from spacy.tokens import Doc
from spacy.attrs import ORTH

class Dong2014(Dataset):

    def __init__(self, rebuild_corpus=False, train_file='train.txt', eval_file='test.txt', debug_file='micro.txt'):
        super().__init__('Dong2014/'+train_file, 'Dong2014/'+eval_file, 'Dong2014/'+debug_file)
        self.generated_data_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'Dong2014', 'generated')
        self.corpus_file = os.path.join(self.generated_data_directory, 'corpus.csv') 
        self.vocabulary_corpus = self.extract_vocabulary_corpus(rebuild=rebuild_corpus)

    def corpus_file_exists(self):
        return os.path.isfile(self.corpus_file)

    def embedding_id_file_exists(self, embedding):
        return os.path.exists(os.path.join(self.generated_data_directory, type(embedding).__name__, embedding.get_alias(), 'words_to_ids.csv'))
    
    def partial_embedding_file_exists(self, embedding):
        return os.path.exists(os.path.join(self.generated_data_directory, type(embedding).__name__, embedding.get_alias(), 'partial_'+embedding.get_version()+'.txt'))
    
    def features_labels_save_file_exists(self, embedding, mode):
        return os.path.exists(os.path.join(self.generated_data_directory, type(embedding).__name__, embedding.get_alias(), 'features_labels_'+mode+'.pkl'))

    def get_features_and_labels(self, mode='debug'):
        file_path = super()._get_file(mode)

        features = {
            'sentence' : [],
            'sentence_length': [],
            'target' : [] 
        }
        labels = []
        
        with open(file_path, "r") as f:
            for line in f:
                if '$T$' in line:
                    features['sentence'].append(line.strip())
                    features['sentence_length'].append(len(line.strip()))
                elif '$T$' not in line and not(re.match(r"^[-10]*$", line)):
                    features['target'].append(line.strip())
                elif re.match(r"^[-10]*$", line.strip()):
                    labels.append(int(line.strip()))

        return features, labels

    def get_mapped_features_and_labels(self, embedding, mode='debug'):
        file_path = super()._get_file(mode)

        if self.partial_embedding_file_exists(embedding):
            embedding.load_embeddings_from_path(path=os.path.join(self.generated_data_directory, type(embedding).__name__, embedding.get_alias(), 'partial_'+embedding.get_version()+'.txt'))
        else:
            embedding.filter_on_corpus(self.vocabulary_corpus['words'])
            partial_embedding_dict = embedding.get_embedding_dictionary()
            os.makedirs(os.path.join(self.generated_data_directory, type(embedding).__name__, embedding.get_alias()), exist_ok=True)
            with open(os.path.join(self.generated_data_directory, type(embedding).__name__, embedding.get_alias(), 'partial_'+embedding.get_version()+'.txt'), 'w+') as f:
                for word in [*partial_embedding_dict]:
                    f.write(word + ' ' + ' '.join(partial_embedding_dict[word].astype(str)) + '\n')
            with open(os.path.join(self.generated_data_directory, type(embedding).__name__, embedding.get_alias(), 'tensorboard_projection_labels.tsv'), 'w+') as f:
                f.write('Words')
                for word in [*partial_embedding_dict]:
                    f.write(word + '\n')

        if self.features_labels_save_file_exists(embedding, mode):
            with open(os.path.join(self.generated_data_directory, type(embedding).__name__, embedding.get_alias(), 'features_labels_'+mode+'.pkl'), 'rb') as f:
                saved_data = pickle.load(f)
                features = saved_data['features']
                labels = saved_data['labels']
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


            token_to_ids_dict = self.extract_embedding_id_file(embedding)
            lines = open(file_path, 'r').readlines()
            for index in range(0, len(lines), 3):
                if (index%15==0):
                    print(index)

                sentence = lines[index].strip()
                features['sentence'].append(sentence)
                features['sentence_length'].append(len(sentence))
                left_context, _, right_context = sentence.partition('$T$')
                features['mappings']['left'].append(embedding.map_embedding_ids(left_context.strip(), token_to_ids_dict=token_to_ids_dict))
                features['mappings']['right'].append(embedding.map_embedding_ids(right_context.strip(), token_to_ids_dict=token_to_ids_dict))

                target = lines[index+1].strip()
                features['target'].append(target)
                features['mappings']['target'].append(embedding.map_embedding_ids(target, token_to_ids_dict=token_to_ids_dict))

                label = lines[index+2].strip()
                labels.append(int(label))
            
            saved_data = {
                'features': features,
                'labels': labels
            }

            with open(os.path.join(self.generated_data_directory, type(embedding).__name__, embedding.get_alias(), 'features_labels_'+mode+'.pkl'), 'wb') as f:
                pickle.dump(saved_data, f, pickle.HIGHEST_PROTOCOL)

        return features, labels

    def extract_vocabulary_corpus(self, rebuild=False):
        vocabulary_corpus = {
            'words': [],
            'counts': []
        }

        if self.corpus_file_exists() and not(rebuild):
            with open(self.corpus_file) as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    vocabulary_corpus['words'].append(row['words'])
                    vocabulary_corpus['counts'].append(row['counts'])
        else:
            all_documents = []
            sentence  = ""
            with fileinput.input(files=(super()._get_file(mode='train'), super()._get_file(mode='eval'))) as f:
                for line in f:
                    if '$T$' in line:
                        sentence = line.strip()
                    elif '$T$' not in line and not(re.match(r"^[-10]*$", line)):
                        all_documents.append(sentence.replace("$T$", line.strip()))
            nlp = spacy.load('en')
            doc = nlp(' '.join(all_documents))
            counts = doc.count_by(ORTH)
            with open(self.corpus_file, 'w') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=['words', 'counts'])

                writer.writeheader()
                for word_id, count in sorted(counts.items(), reverse=True, key=lambda item: item[1]):
                    writer.writerow({'words': nlp.vocab.strings[word_id], 'counts': count})
                    vocabulary_corpus['words'].append(nlp.vocab.strings[word_id])
                    vocabulary_corpus['counts'].append(count)
            
        return vocabulary_corpus
        
    def extract_embedding_id_file(self, embedding):
        token_to_ids_dict = {}
        if self.embedding_id_file_exists(embedding=embedding):
            with open(os.path.join(self.generated_data_directory, type(embedding).__name__, embedding.get_alias(), 'words_to_ids.csv')) as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    token_to_ids_dict[row['word']] = row['id']
        else:
            print('Building token to ids dict...')
            start = time.time()
            token_to_ids_dict = embedding.map_token_ids_dict(self.vocabulary_corpus['words'])
            print('Built token to ids dict in: '+str(time.time()-start))
            os.makedirs(os.path.join(self.generated_data_directory, type(embedding).__name__, embedding.get_alias()), exist_ok=True)
            with open(os.path.join(self.generated_data_directory, type(embedding).__name__, embedding.get_alias(), 'words_to_ids.csv'), 'w+') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=['word', 'id'])

                writer.writeheader()
                for word in [*token_to_ids_dict]:
                    writer.writerow({'word': word, 'id': token_to_ids_dict[word]})
        return token_to_ids_dict

        