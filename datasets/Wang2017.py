import json
import os
import itertools
from collections import defaultdict
from datasets.Dataset import Dataset

class Wang2017(Dataset):

    def __init__(self, train_file='train_id.json', eval_file='test_id.json', debug_file='', parent_folder='', embedding=None, rebuild_corpus=False):
        super().__init__(train_file, eval_file, parent_folder, debug_file, embedding, rebuild_corpus)

    def get_dataset_dictionary(self, mode='debug'):
        dataset_dict = {
            'sentences':[],
            'targets':[],
            'offsets':[],
            'labels':[]
        }

        lines = open(self.get_file(mode), 'r').readlines()
        tweets = []
        annotations = []
        for line in lines:
            tweets.append(json.loads(open(os.path.join(self.parent_directory, 'tweets', '5'+line.strip()+'.json'), 'r').read()))
            annotations.append(json.loads(open(os.path.join(self.parent_directory, 'annotations', '5'+line.strip()+'.json'), 'r').read()))
        
        tweets_and_annotations = defaultdict(lambda: {})
        for sample in itertools.chain(tweets, annotations):
            tweets_and_annotations[sample['tweet_id']].update(sample)

        data = tweets_and_annotations.values()

        dataset_dict['sentences'] = [j for i in [[sample['content']]*len([entity for entity in sample['entities'] if sample['items'][str(entity['id'])] in ['positive','neutral','negative']]) for sample in data] for j in i]
        dataset_dict['targets'] = [j for i in [[entity['entity'] for entity in sample['entities'] if sample['items'][str(entity['id'])] in ['positive','neutral','negative']] for sample in data] for j in i]
        dataset_dict['offsets'] = [j for i in [[entity['offset'] for entity in sample['entities'] if sample['items'][str(entity['id'])] in ['positive','neutral','negative']] for sample in data] for j in i]
        dataset_dict['labels'] = [j for i in [[{'positive': 1, 'neutral': 0, 'negative': -1}.get(sample['items'][str(entity['id'])]) for entity in sample['entities'] if sample['items'][str(entity['id'])] in ['positive','neutral','negative']] for sample in data] for j in i]

        return dataset_dict