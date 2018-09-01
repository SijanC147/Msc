import json
from datasets.Dataset import Dataset

class Saeidi2006(Dataset):

    def __init__(self, train_file='sentihood-train.json', eval_file='sentihood-test.json', debug_file='sentihood-dev.json', parent_folder='', embedding=None, rebuild_corpus=False):
        super().__init__(train_file, eval_file, parent_folder, debug_file, embedding, rebuild_corpus)

    def get_dataset_dictionary(self, mode='debug'):
        dataset_dict = {
            'sentences':[],
            'targets':[],
            'labels':[]
        }
        data = json.loads(open(self.get_file(mode), 'r').read())
        dataset_dict['sentences'] = [j for i in [[sample['text']]*len([opinion for opinion in sample['opinions'] if opinion['aspect']=='general']) for sample in data] for j in i]
        dataset_dict['targets'] = [j for i in [[opinion['target_entity'] for opinion in sample['opinions'] if opinion['aspect']=='general'] for sample in data] for j in i]
        dataset_dict['labels'] = [j for i in [[{'Positive': 1, 'Negative': 0}.get(opinion['sentiment']) for opinion in sample['opinions'] if opinion['aspect']=='general'] for sample in data] for j in i]
        return dataset_dict