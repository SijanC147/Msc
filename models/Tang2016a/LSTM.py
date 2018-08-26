import tensorflow as tf

def eval_input_fn(features, labels, batch_size):
    pass

def train_input_fn(features, labels, batch_size):
    pass

class LSTM:

    def __init__(self, feature_columns, **params):
        self.feature_columns = feature_columns
        self.params = params
        self.train_input_fn = train_input_fn
        self.eval_input_fn = eval_input_fn


    def set_feature_columns(self,feature_columns):
        self.feature_columns = feature_columns

    def get_feature_columns(self):
        return self.feature_columns

    def set_input_fn(self, fn, mode='train'):
        if mode=='train':
            self.train_input_fn = fn
        else:
            self.eval_input_fn = fn

    def get_input_fn(self, fn, mode='train'):
        if mode=='train':
            return self.train_input_fn 
        else:
            return self.eval_input_fn

