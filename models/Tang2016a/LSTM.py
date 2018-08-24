import tensorflow as tf

class LSTM:

    def __init__(self, feature_columns, **params):
        self.feature_columns = feature_columns
        self.params = params


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

