import fileinput
import time
import os
import re
from datasets.Dataset import Dataset

class Dong2014(Dataset):

    def __init__(self, embedding, rebuild_corpus=False, parent_folder='Dong2014', train_file='train.txt', eval_file='test.txt', debug_file='micro.txt'):
        super().__init__(embedding, parent_folder, train_file, eval_file, debug_file)
        self.vocabulary_corpus = self.get_vocabulary_corpus(rebuild=rebuild_corpus)

    def get_vocabulary_corpus(self, rebuild=False):
        if self.corpus_file_exists() and not(rebuild):
            return self.load_vocabulary_corpus_from_csv()
        else:
            all_documents = []
            sentence  = ""
            with fileinput.input(files=(self.train_file_path, self.eval_file_path)) as f:
                for line in f:
                    if '$T$' in line:
                        sentence = line.strip()
                    elif '$T$' not in line and not(re.match(r"^[-10]*$", line)):
                        all_documents.append(sentence.replace("$T$", line.strip()))
        return self.generate_vocabulary_corpus(all_documents)

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

            token_to_ids_dict = self.get_word_to_id_dict(self.vocabulary_corpus)
            file_path = self.get_file(mode)
            lines = open(file_path, 'r').readlines()

            total_time = 0
            for index in range(0, len(lines), 3):

                start = time.time()

                sentence = lines[index].strip()
                features['sentence'].append(sentence)
                features['sentence_length'].append(len(sentence))
                left_context, _, right_context = sentence.partition('$T$')
                features['mappings']['left'].append(self.embedding.map_embedding_ids(left_context.strip(), token_to_ids_dict=token_to_ids_dict))
                features['mappings']['right'].append(self.embedding.map_embedding_ids(right_context.strip(), token_to_ids_dict=token_to_ids_dict))

                target = lines[index+1].strip()
                features['target'].append(target)
                features['mappings']['target'].append(self.embedding.map_embedding_ids(target, token_to_ids_dict=token_to_ids_dict))

                label = lines[index+2].strip()
                labels.append(int(label))

                total_time += (time.time()-start)
                if(index%60==0):
                    print("Processed {0}/{1} lines ({2:.2f}%) tot:{3:.3f}s avg:{4:.3f}s/line".format(index+3, len(lines), ((index+3)/len(lines))*100, total_time, total_time/(index+3)))

            self.save_features_and_labels(mode, features, labels)            

        return features, labels