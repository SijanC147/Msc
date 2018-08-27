import tensorflow as tf
from utils import embed_and_concat


    

class LSTM:
    def __init__(self, **params):
        self.params = params

        def train_input_fn(features, labels, batch_size, max_seq_length=params['max_seq_length'], embedding_dim=params['embedding_dim'], num_out_classes=params['num_out_classes']):
            left_contexts =  tf.data.Dataset.from_generator(lambda: features['mappings']['left'], output_shapes=[None], output_types=tf.int32)
            targets = tf.data.Dataset.from_generator(lambda: features['mappings']['target'], output_shapes=[None], output_types=tf.int32)
            right_contexts = tf.data.Dataset.from_generator(lambda: features['mappings']['right'], output_shapes=[None], output_types=tf.int32)
            sentence_lengths = tf.data.Dataset.from_tensor_slices(features['sentence_length'])

            zipped_features = tf.data.Dataset.zip((left_contexts, targets, right_contexts))
            embedded_features = zipped_features.map(embed_and_concat)

            embedded_features_with_lengths = embedded_features.zip(({'x' : embedded_features, 'len' : sentence_lengths})) 

            one_hot_labels = tf.data.Dataset.from_tensor_slices(tf.one_hot([label+1 for label in labels], depth=3))

            dataset = tf.data.Dataset.zip((embedded_features_with_lengths, one_hot_labels))

            return dataset.shuffle(1000).padded_batch(batch_size=batch_size, padded_shapes=(({'x': [max_seq_length, embedding_dim], 'len': []}), [num_out_classes]))

        def eval_input_fn(features, labels, batch_size, max_seq_length=params['max_seq_length'], embedding_dim=params['embedding_dim'], num_out_classes=params['num_out_classes']):
            train_input_fn(features, labels, batch_size, max_seq_length, embedding_dim, num_out_classes) 

        # def model_fn(features, labels, mode, params=self.params):
        #     inputs = tf.

        #     with tf.name_scope('lstm'):
    
        self.train_input_fn = train_input_fn
        self.eval_input_fn = eval_input_fn
        # self.model_fn = model_fn

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

