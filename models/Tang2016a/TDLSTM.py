import tensorflow as tf
from models.Model import Model
from models.Tang2016a.common import shared_params,shared_feature_columns,tdlstm_input_fn,dual_lstm_model_fn

class TDLSTM(Model):

    def set_params(self, params):
        default_params = shared_params 
        super().set_params(default_params if params==None else params)

    def set_feature_columns(self, feature_columns):
        default_feature_columns = shared_feature_columns  
        super().set_feature_columns(default_feature_columns if feature_columns==None else feature_columns)

    def set_train_input_fn(self, train_input_fn):
        default_train_input_fn = lambda features,labels,batch_size=self.params.get('batch_size'): tdlstm_input_fn(
            features, labels, batch_size, embedding=self.embedding, max_seq_length=self.params['max_seq_length'], num_out_classes=self.params['n_out_classes'])
        super().set_train_input_fn(default_train_input_fn if train_input_fn==None else train_input_fn)        
        
    def set_eval_input_fn(self, eval_input_fn):
        default_eval_input_fn = lambda features,labels,batch_size=self.params.get('batch_size'): tdlstm_input_fn(
            features, labels, batch_size, embedding=self.embedding, max_seq_length=self.params['max_seq_length'], num_out_classes=self.params['n_out_classes'])
        super().set_eval_input_fn(default_eval_input_fn if eval_input_fn==None else eval_input_fn)

    def set_model_fn(self, model_fn):
        default_model_fn = lambda features,labels,mode,params=self.params: dual_lstm_model_fn(
            features, labels, mode, params
        )
        super().set_model_fn(default_model_fn if model_fn==None else model_fn)

