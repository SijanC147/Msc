import tensorflow as tf
from models.Model import Model
from models.Tang2016a.common import tclstm_input_fn,shared_params,shared_feature_columns,shared_lstm_cell_with_dropout

class TCLSTM(Model):

    def set_params(self, params):
        default_params = shared_params 
        super().set_params(default_params if params==None else params)

    def set_feature_columns(self, feature_columns):
        default_feature_columns = shared_feature_columns  
        super().set_feature_columns(default_feature_columns if feature_columns==None else feature_columns)

    def set_train_input_fn(self, train_input_fn):
        default_train_input_fn = lambda features,labels,batch_size=self.params.get('batch_size'): tclstm_input_fn(
            features, labels, batch_size, max_seq_length=self.params['max_seq_length'])
        super().set_train_input_fn(default_train_input_fn if train_input_fn==None else train_input_fn)        
        
    def set_eval_input_fn(self, eval_input_fn):
        default_eval_input_fn = lambda features,labels,batch_size=self.params.get('batch_size'): tclstm_input_fn(
            features, labels, batch_size, max_seq_length=self.params['max_seq_length'],eval_input=True)
        super().set_eval_input_fn(default_eval_input_fn if eval_input_fn==None else eval_input_fn)

    def set_model_fn(self, model_fn):
        def default_model_fn(features, labels, mode, params=self.params):
            with tf.variable_scope('embedding_layer', reuse=tf.AUTO_REUSE):
                embeddings = tf.get_variable(
                    'embeddings', 
                    shape=[params['vocab_size'],params['embedding_dim']],
                    initializer=params['embedding_initializer'])
            
            target_embedding = tf.contrib.layers.embed_sequence(
                ids=features['target']['x'], 
                initializer=embeddings,
                scope='embedding_layer',
                reuse=True
            )

            left_inputs = tf.contrib.layers.embed_sequence(
                ids=features['left']['x'], 
                initializer=embeddings,
                scope='embedding_layer',
                reuse=True
            )

            right_inputs = tf.contrib.layers.embed_sequence(
                ids=features['right']['x'], 
                initializer=embeddings,
                scope='embedding_layer',
                reuse=True
            )

            with tf.name_scope('target_connection'):
                mean_target_embedding = tf.reduce_mean(
                        input_tensor=target_embedding[:,:features['target']['len'][0],:],
                        axis=1,
                        keepdims=True)
                left_inputs = tf.stack(
                    # values=[left_inputs,tf.ones([tf.shape(left_inputs)[0],params['max_seq_length'],params['embedding_dim']])*mean_target_embedding],
                    values=[left_inputs,tf.ones(tf.shape(left_inputs))*mean_target_embedding],
                    axis=2
                )
                left_inputs = tf.reshape(
                    tensor=left_inputs, 
                    shape=[-1,params['max_seq_length'],2*params['embedding_dim']])
                right_inputs = tf.stack(
                    # values=[right_inputs,tf.ones([tf.shape(right_inputs)[0],params['max_seq_length'],params['embedding_dim']])*mean_target_embedding],
                    values=[right_inputs,tf.ones(tf.shape(right_inputs))*mean_target_embedding],
                    axis=2
                )
                right_inputs = tf.reshape(
                    tensor=right_inputs, 
                    shape=[-1,params['max_seq_length'],2*params['embedding_dim']])

            
            with tf.variable_scope('left_lstm'):
                _, final_states_left = tf.nn.dynamic_rnn(
                    cell=shared_lstm_cell_with_dropout(params),
                    inputs=left_inputs,
                    sequence_length=features['left']['len'],
                    dtype=tf.float32
                )
            
            with tf.variable_scope('right_lstm'):
                _, final_states_right = tf.nn.dynamic_rnn(
                    cell=shared_lstm_cell_with_dropout(params),
                    inputs=right_inputs,
                    sequence_length=features['right']['len'],
                    dtype=tf.float32
                )
            
            concatenated_final_states = tf.concat([final_states_left.h, final_states_right.h], axis=1)

            logits = tf.layers.dense(
                inputs=concatenated_final_states, 
                units=params['n_out_classes'])

            predicted_classes = tf.argmax(logits, 1)

            if mode == tf.estimator.ModeKeys.PREDICT:
                predictions = {
                    'class_ids': predicted_classes[:, tf.newaxis],
                    'probabilities': tf.nn.softmax(logits),
                    'logits': logits
                }
                return tf.estimator.EstimatorSpec(mode, predictions=predictions)
            
            loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
            accuracy = tf.metrics.accuracy(labels=labels, predictions=predicted_classes)

            metrics = {
                'accuracy': accuracy
            }

            tf.summary.scalar('accuracy', accuracy[1])
            tf.summary.scalar('loss', loss)

            if mode == tf.estimator.ModeKeys.EVAL:
                return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

            optimizer = tf.train.AdagradOptimizer(learning_rate=params['learning_rate'])
            train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

            logging_hook = tf.train.LoggingTensorHook({'loss': loss, 'accuracy': accuracy[1]}, every_n_iter=50)

            return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op, training_hooks=[logging_hook])
    
        super().set_model_fn(default_model_fn if model_fn==None else model_fn)