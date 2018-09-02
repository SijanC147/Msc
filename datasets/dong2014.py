from datasets.Dataset import Dataset

class Dong2014(Dataset):

    def __init__(self, train_file='train.txt', eval_file='test.txt', debug_file='micro.txt', embedding=None, rebuild_corpus=False):
        super().__init__(train_file_path=train_file, eval_file_path=eval_file, debug_file_path=debug_file, embedding=embedding, rebuild_corpus=rebuild_corpus)

    def generate_dataset_dictionary(self, mode='debug'):
        dataset_dict = {
            'sentences':[],
            'targets':[],
            'labels':[]
        }
        lines = open(self.get_file(mode), 'r').readlines()
        dataset_dict['sentences'] = [lines[index].strip().replace('$T$', lines[index+1].strip()) for index in range(0, len(lines), 3)]
        dataset_dict['targets'] = [lines[index+1].strip() for index in range(0, len(lines), 3)]
        dataset_dict['labels'] = [int(lines[index+2].strip()) for index in range(0, len(lines), 3)]
        return dataset_dict
