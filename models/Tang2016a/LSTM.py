import tensorflow as tf
from models.Model import Model
from models.Tang2016a.common import lstm_input_fn,shared_params,shared_feature_columns

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
        default_eval_input_fn = lambda features,labels,batch_size=self.params.get('batch_size'): lstm_input_fn(
            features, labels, batch_size, embedding=self.embedding, max_seq_length=self.params['max_seq_length'], num_out_classes=self.params['n_out_classes'])
        super().set_eval_input_fn(default_eval_input_fn if eval_input_fn==None else eval_input_fn)

    def set_model_fn(self, model_fn):
        def default_model_fn(features, labels, mode, params=self.params):
            print(features)
            input_layer, sequence_length = tf.contrib.feature_column.sequence_input_layer(
                features=features,
                feature_columns=params['feature_columns']
            )

            lstm_cell = tf.nn.rnn_cell.LSTMCell(params['hidden_units'])

            _, final_states = tf.nn.dynamic_rnn(
                cell=lstm_cell,
                inputs=input_layer,
                sequence_length=sequence_length,
                dtype=tf.float32
            )

            logits = tf.layers.dense(inputs=final_states.h, units=params['n_out_classes'], trainable=False)
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
            # recall = tf.metrics.recall(labels=labels, predictions=predicted_classes, name='recall_op')

            metrics = {
                'accuracy': accuracy,
                # 'recall': recall
                }

            tf.summary.scalar('accuracy', accuracy[1])
            # tf.summary.scalar('recall', recall[1])

            tf.summary.scalar('loss', loss)

            if mode == tf.estimator.ModeKeys.EVAL:
                return tf.estimator.EstimatorSpec(
                    mode, loss=loss, eval_metric_ops=metrics)
            
            optimizer = tf.train.AdagradOptimizer(learning_rate=params['learning_rate'])

            train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

            return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
    
        super().set_model_fn(default_model_fn if model_fn==None else model_fn)

