import os
import time
import tensorflow as tf
from utils import embed_and_concat,get_embedding_matrix_variable

class LSTM:
    def __init__(
        self, 
        feature_columns, 
        embedding,
        model_dir = '',
        max_seq_length = 80, 
        n_out_classes = 3, 
        learning_rate = 0.01,
        dropout_rate = 0.1,
        hidden_units = 200
    ):
        self.set_model_dir(model_dir=model_dir)

        self.params = {
            'feature_columns' : feature_columns,
            'max_seq_length' : max_seq_length,
            'embedding_dim' : embedding.get_embedding_dim(),
            'n_out_classes' : n_out_classes,
            'learning_rate' : learning_rate,
            'dropout_rate' : dropout_rate,
            'hidden_units' : hidden_units
        }

        def train_input_fn(features, labels, batch_size, max_seq_length=self.params['max_seq_length'], embedding_dim=self.params['embedding_dim'], num_out_classes=self.params['n_out_classes']):
            embedding.set_embedding_matrix_variable()

            left_contexts =  tf.data.Dataset.from_generator(lambda: features['mappings']['left'], output_shapes=[None], output_types=tf.int32)
            targets = tf.data.Dataset.from_generator(lambda: features['mappings']['target'], output_shapes=[None], output_types=tf.int32)
            right_contexts = tf.data.Dataset.from_generator(lambda: features['mappings']['right'], output_shapes=[None], output_types=tf.int32)

            zipped_features = tf.data.Dataset.zip((left_contexts, targets, right_contexts))
            embedded_features = zipped_features.map(embed_and_concat)
            sparse_features = embedded_features.apply(tf.contrib.data.dense_to_sparse_batch(batch_size=1, row_shape=[max_seq_length,embedding_dim]))
            sparse_features_dict = tf.data.Dataset.zip(({'x' : sparse_features})) 

            labels_dataset = tf.data.Dataset.from_tensor_slices([label+1 for label in labels])

            dataset = tf.data.Dataset.zip((sparse_features_dict, labels_dataset))

            return dataset.shuffle(len(features['sentence'])).repeat().batch(batch_size=batch_size)

        def eval_input_fn(features, labels, batch_size, max_seq_length=self.params['max_seq_length'], embedding_dim=self.params['embedding_dim'], num_out_classes=self.params['n_out_classes']):
            train_input_fn(features, labels, batch_size, max_seq_length, embedding_dim, num_out_classes) 

        def model_fn(features, labels, mode, params=self.params):
            input_layer, sequence_length = tf.contrib.feature_column.sequence_input_layer(
                features=features,
                feature_columns=params['feature_columns']
            )

            lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(params['hidden_units'])

            _, final_states = tf.nn.dynamic_rnn(
                cell=lstm_cell,
                inputs=input_layer,
                sequence_length=sequence_length,
                dtype=tf.float32
            )

            logits = tf.layers.dense(inputs=final_states.h, units=params['n_out_classes'])
            predicted_classes = tf.argmax(logits, 1)

            if mode == tf.estimator.ModeKeys.PREDICT:
                predictions = {
                    'class_ids': predicted_classes[:, tf.newaxis],
                    'probabilities': tf.nn.softmax(logits),
                    'logits': logits
                }
                return tf.estimator.EstimatorSpec(mode, predictions=predictions)

            loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

            accuracy = tf.metrics.accuracy(labels=labels,predictions=predicted_classes,name='acc_op')
            recall = tf.metrics.recall(labels=labels, predictions=predicted_classes, name='recall_op')

            metrics = {
                'accuracy': accuracy,
                'recall': recall
                }

            tf.summary.scalar('accuracy', accuracy[1])
            tf.summary.scalar('recall', recall[1])

            tf.summary.scalar('loss', loss)

            if mode == tf.estimator.ModeKeys.EVAL:
                return tf.estimator.EstimatorSpec(
                    mode, loss=loss, eval_metric_ops=metrics)
            
            optimizer = tf.train.AdagradOptimizer(learning_rate=params['learning_rate'])

            train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

            return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
    
        self.train_input_fn = train_input_fn
        self.eval_input_fn = eval_input_fn
        self.model_fn = model_fn

    def set_feature_columns(self,feature_columns):
        self.params['feature_columns'] = feature_columns

    def get_feature_columns(self):
        return self.params['feature_columns']

    def get_model_dir(self):
        return self.model_dir

    def set_model_dir(self, model_dir):
        specific_model_folder = os.path.basename(os.path.normpath(os.path.splitext(os.path.abspath(__file__))[0]))
        specific_model_dir = os.path.join(os.getcwd(), 'logs', os.path.relpath(os.path.dirname(os.path.abspath(__file__)), os.path.join(os.getcwd(), 'models')), specific_model_folder)
        self.model_dir = os.path.join(specific_model_dir, model_dir.replace(" ", "_")) if len(model_dir)>0 else os.path.join(specific_model_dir, '_default')

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

