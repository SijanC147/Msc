import json
from datasets.Dataset import Dataset

class Xue2018(Dataset):

    def __init__(self, version='all_joined', train_file='train.json', eval_file='test.json', debug_file='', parent_folder='', embedding=None, rebuild_corpus=False):
        parent_folder = {
            'all': 'all_joined',
            'easy': 'easy_joined',
            'hard': 'hard_joined',
            'laptop-easy': 'laptop_easy',
            'laptop-hard': 'laptop_hard',
            'laptop': 'laptop_joined',
            'restaurant-easy': 'restaurant_easy',
            'restaurant-hard': 'restaurant_hard',
            'restaurant': 'restaurant_joined',
        }.get(version, 'all_joined')
        super().__init__(train_file, eval_file, parent_folder, debug_file, embedding, rebuild_corpus)

    def generate_dataset_dictionary(self, mode):
        dataset_dict = {
            'sentences':[],
            'targets':[],
            'labels':[]
        }
        data = json.loads(open(self.get_file(mode), 'r').read())
        data = [sample for sample in data if sample['aspect']!='conflict']
        dataset_dict['sentences'] = [sample['sentence'] for sample in data if sample['sentiment'] in ['positive','neutral','negative']]
        dataset_dict['targets'] = [sample['aspect'] for sample in data if sample['sentiment'] in ['positive','neutral','negative']]
        dataset_dict['labels'] = [{'positive': 1, 'neutral': 0, 'negative': -1}.get(sample['sentiment']) for sample in data if sample['sentiment'] in ['positive','neutral','negative']]
        return dataset_dict