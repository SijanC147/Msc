import csv
from datasets.Dataset import Dataset

class Nakov2016(Dataset):

    def __init__(self, train_file='twitter-2016train-CE.tsv', eval_file='twitter-2016test-CE.tsv', debug_file='', parent_folder='', embedding=None, rebuild_corpus=False):
        super().__init__(train_file, eval_file, parent_folder, debug_file, embedding, rebuild_corpus)

    def generate_dataset_dictionary(self, mode):
        dataset_dict = {
            'sentences':[],
            'targets':[],
            'offsets':[],
            'labels':[]
        }

        with open(self.get_file(mode), 'r') as f:
            reader = csv.DictReader(f, dialect='excel-tab', fieldnames=['tweet_id', 'target', 'sentiment', 'sentence'])
            for row in reader:
                if row['sentiment'] in ['positive', 'neutral', 'negative']:
                    dataset_dict['sentences'].append(row['sentence'])
                    dataset_dict['target'].append(row['target'])
                    dataset_dict['labels'].append({'2' : 1, '1' : 1, '0': 0, '-1': -1, '-2': -1}.get(row['sentiment']))

        return dataset_dict