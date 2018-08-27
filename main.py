import tensorflow as tf
tf.enable_eager_execution()
import numpy as np

from datasets.Dong2014 import Dong2014
from embeddings.GloVe import GloVe
from utils import embed_and_concat

glove = GloVe('twitterMicro')
dong = Dong2014()

features, labels = dong.get_mapped_features_and_labels(embedding=glove)

feature_columns = [
    tf.contrib.feature_column.sequence_numeric_column(key='x')
]

params = {
    'feature_columns' : feature_columns,
    'max_seq_length' : 40,
    'embedding_dim' : 25,
    'n_out_classes' : 3,
    'learning_rate' : 0.1,
    'dropout_rate' : 0.1,
    'hidden_units' : 100
}

def train_input_fn(features, labels, batch_size):
    left_contexts =  tf.data.Dataset.from_generator(lambda: features['mappings']['left'], output_shapes=[None], output_types=tf.int32)
    targets = tf.data.Dataset.from_generator(lambda: features['mappings']['target'], output_shapes=[None], output_types=tf.int32)
    right_contexts = tf.data.Dataset.from_generator(lambda: features['mappings']['right'], output_shapes=[None], output_types=tf.int32)
    sentence_lengths = tf.data.Dataset.from_tensor_slices(features['sentence_length'])

    zipped_features = tf.data.Dataset.zip((left_contexts, targets, right_contexts))
    embedded_features = zipped_features.map(embed_and_concat)
    sparse_features = embedded_features.apply(tf.contrib.data.dense_to_sparse_batch(batch_size=1, row_shape=[params['max_seq_length'],params['embedding_dim']]))

    sparse_features_with_lengths = tf.data.Dataset.zip(({'x' : sparse_features, 'len' : sentence_lengths})) 

    # one_hot_labels = tf.data.Dataset.from_tensor_slices(tf.one_hot([label+1 for label in labels], depth=params['n_out_classes']))
    labels_dataset = tf.data.Dataset.from_tensor_slices([label+1 for label in labels])

    dataset = tf.data.Dataset.zip((sparse_features_with_lengths, labels_dataset))

    return dataset.shuffle(1000).repeat(10).batch(batch_size=batch_size)

def model_fn(features, labels, mode, params=params):

    initial_value = np.random.randn(27, 25).astype(np.float32)
    embedding_matrix = tf.get_variable("embedding_matrix", [27,25], initializer=tf.constant_initializer(initial_value))

    def init_embedding_matrix_fn(scaffold, session):
        session.run(embedding_matrix.initializer)
        print('got here')
    scaffold = tf.train.Scaffold(init_fn=init_embedding_matrix_fn)

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

classifier = tf.estimator.Estimator(model_fn=model_fn, params=params)
classifier.train(input_fn=lambda: train_input_fn(features, labels, 1))