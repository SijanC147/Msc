import tensorflow as tf
from models.Model import Model
from models.Tang2016a.common import lstm_input_fn,shared_params,shared_feature_columns,shared_lstm_cell,shared_lstm_cell_with_dropout
from utils import random_input_fn

class LSTM(Model):

    def set_params(self, params):
        default_params = shared_params
        super().set_params(default_params if params==None else params)

    def set_feature_columns(self, feature_columns):
        default_feature_columns = shared_feature_columns 
        super().set_feature_columns(default_feature_columns if feature_columns==None else feature_columns)

    def set_train_input_fn(self, train_input_fn):
        default_train_input_fn = lambda features,labels,batch_size=self.params.get('batch_size'): lstm_input_fn(
            features, labels, batch_size, embedding=self.embedding, max_seq_length=self.params['max_seq_length'], num_out_classes=self.params['n_out_classes'])
        super().set_train_input_fn(default_train_input_fn if train_input_fn==None else train_input_fn)        
        
    def set_eval_input_fn(self, eval_input_fn):
        default_eval_input_fn = lambda features,labels: lstm_input_fn(
            features, labels, batch_size=None, embedding=self.embedding, max_seq_length=self.params['max_seq_length'], num_out_classes=self.params['n_out_classes'])
        super().set_eval_input_fn(default_eval_input_fn if eval_input_fn==None else eval_input_fn)

    def set_model_fn(self, model_fn):
        def default_model_fn(features, labels, mode, params=self.params):
            features['x'] = tf.contrib.layers.dense_to_sparse(features['x'])

            input_layer, sequence_length = tf.contrib.feature_column.sequence_input_layer(
                features=features,
                feature_columns=params['feature_columns'],
            )

            sequence_length = tf.Print(input_=sequence_length, data=[sequence_length], message='Seq length: ', name='sequence_length')

            _, final_states = tf.nn.dynamic_rnn(
                cell=tf.nn.rnn_cell.LSTMCell(num_units=params['hidden_units'], initializer=tf.zeros_initializer),
                inputs=input_layer,
                sequence_length=sequence_length,
                dtype=tf.float32
            )

            logits = tf.layers.dense(
                inputs=final_states.h, 
                units=params['n_out_classes'],
                kernel_initializer=tf.zeros_initializer,
                bias_initializer=tf.zeros_initializer)

            labels = tf.Print(input_=labels, data=[labels], message='Labels: ', name='labels')
            logits = tf.Print(input_=logits, data=[logits], message='Logits: ', name='logits')
            predicted_classes = tf.argmax(logits, 1)

            if mode == tf.estimator.ModeKeys.PREDICT:
                predictions = {
                    'class_ids': predicted_classes[:, tf.newaxis],
                    'probabilities': tf.nn.softmax(logits),
                    'logits': logits
                }
                return tf.estimator.EstimatorSpec(mode, predictions=predictions)
            
            loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
            accuracy = tf.metrics.accuracy(labels=labels, predictions=predicted_classes, name='accuracy')

            # metrics = {
            #     'accuracy': accuracy
            # }

            # tf.summary.scalar('accuracy', accuracy[1])
            # tf.summary.scalar('loss', loss)

            if mode == tf.estimator.ModeKeys.EVAL:
                return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops={})

            # optimizer = tf.train.AdagradOptimizer(learning_rate=params['learning_rate'])
            optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate'])
            train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

            logging_hook = tf.train.LoggingTensorHook({'loss': loss, 'accuracy': accuracy[1]}, every_n_iter=50)

            return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op, training_hooks=[logging_hook])
    
        super().set_model_fn(default_model_fn if model_fn==None else model_fn)

