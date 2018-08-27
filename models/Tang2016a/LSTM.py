import tensorflow as tf
from utils import embed_and_concat,get_embedding_matrix_variable

class LSTM:
    def __init__(
        self, 
        feature_columns, 
        embedding,
        max_seq_length = 40, 
        embedding_dim = 25, 
        n_out_classes = 3, 
        learning_rate = 0.1,
        dropout_rate = 0.1,
        hidden_units = 100
    ):
        self.params = {
            'feature_columns' : feature_columns,
            'max_seq_length' : max_seq_length,
            'embedding_dim' : embedding_dim,
            'n_out_classes' : n_out_classes,
            'learning_rate' : learning_rate,
            'dropout_rate' : dropout_rate,
            'hidden_units' : hidden_units
        }

        def train_input_fn(features, labels, batch_size, max_seq_length=self.params['max_seq_length'], embedding_dim=self.params['embedding_dim'], num_out_classes=self.params['n_out_classes']):
            left_contexts =  tf.data.Dataset.from_generator(lambda: features['mappings']['left'], output_shapes=[None], output_types=tf.int32)
            targets = tf.data.Dataset.from_generator(lambda: features['mappings']['target'], output_shapes=[None], output_types=tf.int32)
            right_contexts = tf.data.Dataset.from_generator(lambda: features['mappings']['right'], output_shapes=[None], output_types=tf.int32)
            sentence_lengths = tf.data.Dataset.from_tensor_slices(features['sentence_length'])

            zipped_features = tf.data.Dataset.zip((left_contexts, targets, right_contexts))
            embedded_features = zipped_features.map(embed_and_concat)

            embedded_features_with_lengths = embedded_features.zip(({'x' : embedded_features, 'len' : sentence_lengths})) 

            one_hot_labels = tf.data.Dataset.from_tensor_slices(tf.one_hot([label+1 for label in labels], depth=3))

            dataset = tf.data.Dataset.zip((embedded_features_with_lengths, one_hot_labels))

            return dataset.shuffle(1000).repeat(10).padded_batch(batch_size=batch_size, padded_shapes=(({'x': [max_seq_length, embedding_dim], 'len': []}), [num_out_classes]))

        def eval_input_fn(features, labels, batch_size, max_seq_length=self.params['max_seq_length'], embedding_dim=self.params['embedding_dim'], num_out_classes=self.params['n_out_classes']):
            train_input_fn(features, labels, batch_size, max_seq_length, embedding_dim, num_out_classes) 

        def model_fn(features, labels, mode, params=self.params):
            print("Eager execution: {}".format(tf.executing_eagerly()))

            embedding_matrix = embedding.set_embedding_matrix_variable()
            print(embedding_matrix)

            def init_embedding_matrix_fn(scaffold, session):
                session.run(embedding_matrix.initializer, {embedding_matrix.initial_value: embedding.get_embedding_vectors()})
                print('got here')
            scaffold = tf.train.Scaffold(init_fn=init_embedding_matrix_fn)

            print(embedding_matrix)
            print(next(iter(features)))

            input_layer = tf.contrib.feature_column.sequence_input_layer(
                features=features,
                feature_columns=params['feature_columns']
            )

            lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(params['hidden_units'])

            _, final_states = tf.nn.dynamic_rnn(
                cell=lstm_cell,
                inputs=input_layer,
                sequence_length=features['len'],
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
                return tf.estimator.EstimatorSpec(mode, predictions=predictions, scaffold=scaffold)

            loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

            accuracy = tf.metrics.accuracy(labels=labels,predictions=predicted_classes,name='acc_op')

            metrics = {'accuracy': accuracy}
            tf.summary.scalar('accuracy', accuracy[1])

            if mode == tf.estimator.ModeKeys.EVAL:
                return tf.estimator.EstimatorSpec(
                    mode, loss=loss, eval_metric_ops=metrics, scaffold=scaffold)
            
            optimizer = tf.train.AdagradOptimizer(learning_rate=params['learning_rate'])

            train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

            return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op, scaffold=scaffold)
    
        self.train_input_fn = train_input_fn
        self.eval_input_fn = eval_input_fn
        self.model_fn = model_fn

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

