import tensorflow as tf
from utils import embed_and_concat

def train_input_fn(features, labels, batch_size, max_seq_length):
   left_contexts =  tf.data.Dataset.from_generator(lambda: features['mappings']['left'], output_shapes=[None], output_types=tf.int32)
   targets = tf.data.Dataset.from_generator(lambda: features['mappings']['target'], output_shapes=[None], output_types=tf.int32)
   right_contexts = tf.data.Dataset.from_generator(lambda: features['mappings']['right'], output_shapes=[None], output_types=tf.int32)

   zipped_features = tf.data.Dataset.zip((left_contexts, targets, right_contexts))

   features_dataset = zipped_features.map(embed_and_concat)

   labels_dataset = tf.data.Dataset.from_tensor_slices(tf.one_hot([label+1 for label in labels], depth=3))

   dataset = tf.data.Dataset.zip((features_dataset, labels_dataset))

   return dataset.shuffle(1000).padded_batch(batch_size=batch_size, padded_shapes=([max_seq_length, None],[None]))

def eval_input_fn(features, labels, batch_size, max_seq_length):
   train_input_fn(features, labels, batch_size, max_seq_length) 

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

    def get_input_fn(self, mode='train'):
        if mode=='train':
            return self.train_input_fn 
        else:
            return self.eval_input_fn

