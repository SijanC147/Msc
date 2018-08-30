import os
import time
import tensorflow as tf
from models.Model import Model
from utils import embed_and_concat,get_embedding_matrix_variable

class LSTM(Model):
    def __init__(
        self, 
        max_seq_length = 80, 
        n_out_classes = 3, 
        learning_rate = 0.01,
        dropout_rate = 0.1,
        hidden_units = 200
    ):
        self.max_seq_length = max_seq_length
        self.n_out_classes = n_out_classes
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.hidden_units = hidden_units
        super().__init__()

    def set_params(self, params):
        default_params = {
            'max_seq_length' : self.max_seq_length,
            'n_out_classes' : self.n_out_classes,
            'learning_rate' : self.learning_rate,
            'dropout_rate' : self.dropout_rate,
            'hidden_units' : self.hidden_units
        } 
        super().set_params(default_params if params==None else params)

    def set_feature_columns(self, feature_columns):
        default_feature_columns = [
            tf.contrib.feature_column.sequence_numeric_column(key='x')
        ]  
        super().set_feature_columns(default_feature_columns if feature_columns==None else feature_columns)

    def set_train_input_fn(self, train_input_fn):
        def default_train_input_fn(features, labels, batch_size, embedding=self.embedding, max_seq_length=self.params['max_seq_length'], num_out_classes=self.params['n_out_classes']):
            embedding.set_embedding_matrix_variable()

            left_contexts =  tf.data.Dataset.from_generator(lambda: features['mappings']['left'], output_shapes=[None], output_types=tf.int32)
            targets = tf.data.Dataset.from_generator(lambda: features['mappings']['target'], output_shapes=[None], output_types=tf.int32)
            right_contexts = tf.data.Dataset.from_generator(lambda: features['mappings']['right'], output_shapes=[None], output_types=tf.int32)

            zipped_features = tf.data.Dataset.zip((left_contexts, targets, right_contexts))
            embedded_features = zipped_features.map(embed_and_concat)
            sparse_features = embedded_features.apply(tf.contrib.data.dense_to_sparse_batch(batch_size=1, row_shape=[max_seq_length,embedding.get_embedding_dim()]))
            sparse_features_dict = tf.data.Dataset.zip(({'x' : sparse_features})) 

            labels_dataset = tf.data.Dataset.from_tensor_slices([label+1 for label in labels])

            dataset = tf.data.Dataset.zip((sparse_features_dict, labels_dataset))

            return dataset.shuffle(len(features['sentence'])).repeat().batch(batch_size=batch_size)
        super().set_train_input_fn(default_train_input_fn if train_input_fn==None else train_input_fn)        
        
    def set_eval_input_fn(self, eval_input_fn):
        def default_eval_input_fn(features, labels, batch_size, embedding=self.embedding, max_seq_length=self.params['max_seq_length'], num_out_classes=self.params['n_out_classes']):
            embedding.set_embedding_matrix_variable()

            left_contexts =  tf.data.Dataset.from_generator(lambda: features['mappings']['left'], output_shapes=[None], output_types=tf.int32)
            targets = tf.data.Dataset.from_generator(lambda: features['mappings']['target'], output_shapes=[None], output_types=tf.int32)
            right_contexts = tf.data.Dataset.from_generator(lambda: features['mappings']['right'], output_shapes=[None], output_types=tf.int32)

            zipped_features = tf.data.Dataset.zip((left_contexts, targets, right_contexts))
            embedded_features = zipped_features.map(embed_and_concat)
            sparse_features = embedded_features.apply(tf.contrib.data.dense_to_sparse_batch(batch_size=1, row_shape=[max_seq_length,embedding.get_embedding_dim()]))
            sparse_features_dict = tf.data.Dataset.zip(({'x' : sparse_features})) 

            labels_dataset = tf.data.Dataset.from_tensor_slices([label+1 for label in labels])

            dataset = tf.data.Dataset.zip((sparse_features_dict, labels_dataset))

            return dataset.shuffle(len(features['sentence'])).repeat().batch(batch_size=batch_size)
        super().set_eval_input_fn(default_eval_input_fn if eval_input_fn==None else eval_input_fn)

    def set_model_fn(self, model_fn):
        def default_model_fn(features, labels, mode, params=self.params):
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
    
        super().set_model_fn(default_model_fn if model_fn==None else model_fn)

