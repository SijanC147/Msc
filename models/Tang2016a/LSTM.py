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
            head = tf.contrib.estimator.multi_class_head(n_classes=params['n_out_classes'])

            input_layer, sequence_length = tf.contrib.feature_column.sequence_input_layer(
                features=features,
                feature_columns=params['feature_columns'],
            )

            _, final_states = tf.nn.dynamic_rnn(
                cell=shared_lstm_cell_with_dropout(params),
                inputs=input_layer,
                sequence_length=sequence_length,
                dtype=tf.float32
            )

            logits = tf.layers.dense(
                inputs=final_states.h, 
                units=params['n_out_classes'],
                kernel_initializer=tf.random_uniform_initializer(minval=-0.03,maxval=0.03),
                bias_initializer=tf.random_uniform_initializer(minval=-0.03,maxval=0.03),
                activation=None)

            if labels is not None:
                labels = tf.reshape(labels, [-1,1])

            optimizer = tf.train.AdagradOptimizer(learning_rate=params['learning_rate'])

            tf.summary.scalar('training_accuracy', tf.metrics.accuracy(labels=labels, predictions=tf.argmax(logits,1))[1])

            def _train_op_fn(loss):
                return optimizer.minimize(
                    loss=loss,
                    global_step=tf.train.get_global_step())

            return head.create_estimator_spec(
                features=features,
                labels=labels,
                mode=mode,
                logits=logits,
                train_op_fn=_train_op_fn)
    
        super().set_model_fn(default_model_fn if model_fn==None else model_fn)

