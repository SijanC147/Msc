from abc import ABC, abstractmethod

class Dataset(ABC):

    def __init__(self, train_file_path, eval_file_path, debug_file_path=""):
        self.train_file_path = 'datasets/data/'+train_file_path
        self.eval_file_path = 'datasets/data/'+eval_file_path
        self.debug_file_path = 'datasets/data/'+debug_file_path

    @abstractmethod    
    def get_features_and_labels(self, mode='debug'):
        pass

    @abstractmethod
    def get_mapped_features_and_labels(self, embedding, mode='debug'):
        pass

    def _get_file(self, mode="debug"):
        if mode=='train':
            return self.train_file_path
        elif mode=='eval':
            return self.eval_file_path
        else:
            return self.debug_file_path