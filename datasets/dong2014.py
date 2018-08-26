from datasets.Dataset import Dataset
import re

class Dong2014(Dataset):

    def __init__(self, train_file='train.txt', eval_file='test.txt', debug_file='micro.txt'):
        super().__init__('Dong2014/'+train_file, 'Dong2014/'+eval_file, 'Dong2014/'+debug_file)
    
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

        features = {
            'sentence' : [],
            'sentence_length': [],
            'target' : [],
            'mappings': {
                'left': [],
                'target': [],
                'right': []
            }
        }
        labels = []
        
        with open(file_path, "r") as f:
            for line in f:
                if '$T$' in line:
                    features['sentence'].append(line.strip())
                    features['sentence_length'].append(len(line.strip()))
                    left_mappings, right_mappings = embedding.map_embedding_ids(line.strip(), separate_on="$T$")
                    features['mappings']['left'].append(left_mappings)
                    features['mappings']['right'].append(right_mappings)
                elif '$T$' not in line and not(re.match(r"^[-10]*$", line)):
                    features['target'].append(line.strip())
                    features['mappings']['target'].append(embedding.map_embedding_ids(line.strip()))
                elif re.match(r"^[-10]*$", line.strip()):
                    labels.append(int(line.strip()))

        return features, labels
