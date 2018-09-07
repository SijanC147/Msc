import tensorflow as tf
from tensorflow.python.keras.preprocessing import sequence # pylint: disable=E0611

from utils import embed_and_concat, embed_from_ids, embed_target_and_average

shared_params = {
    'batch_size': 200,
    'max_seq_length' : 85, 
    'n_out_classes' : 3, 
    'learning_rate' : 0.01,
    'keep_prob' : 0.8,
    'hidden_units' : 200,
    }

shared_feature_columns = [
    tf.contrib.feature_column.sequence_numeric_column(key='x')
    ]

shared_lstm_cell = lambda params: tf.nn.rnn_cell.LSTMCell(num_units=params['hidden_units'], initializer=tf.initializers.random_uniform(minval=-0.03, maxval=0.03))

shared_lstm_cell_with_dropout = lambda params: tf.contrib.rnn.DropoutWrapper(cell=shared_lstm_cell(params), output_keep_prob=params['keep_prob'])

def lstm_input_fn(features, labels, batch_size, max_seq_length):
    sentences = [l+t+r for l,t,r in zip(features['mappings']['left'],features['mappings']['target'],features['mappings']['right'])]
    sens_lens = [len(l+t+r) for l,t,r in zip(features['mappings']['left'],features['mappings']['target'],features['mappings']['right'])]
    labels = [label+1 for label in labels]

    sentences = sequence.pad_sequences(sequences=sentences, maxlen=max_seq_length, truncating='post', padding='post', value=0)
    
    dataset = tf.data.Dataset.from_tensor_slices((sentences, sens_lens, labels))
    dataset = dataset.shuffle(buffer_size=len(labels))

    if batch_size==None:
        dataset = dataset.batch(batch_size=1)
    else:
        dataset = dataset.batch(batch_size=batch_size)

    dataset = dataset.map(lambda sentence,length,label: ({'x': sentence, 'len': length}, label))

    if batch_size!=None:
        dataset = dataset.repeat()

    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()

def tdlstm_input_fn(features, labels, batch_size, max_seq_length):
    left_contexts = [l+t for l,t in zip(features['mappings']['left'],features['mappings']['target'])]
    left_contexts_len = [len(left_context) for left_context in left_contexts]
    left_contexts = sequence.pad_sequences(sequences=left_contexts, maxlen=max_seq_length, truncating='post', padding='post', value=0)

    right_contexts = [list(reversed(t+r)) for t,r in zip(features['mappings']['target'],features['mappings']['right'])]
    right_contexts_len = [len(right_context) for right_context in right_contexts]
    right_contexts = sequence.pad_sequences(sequences=right_contexts, maxlen=max_seq_length, truncating='post', padding='post', value=0)

    labels = [label+1 for label in labels]
    
    dataset = tf.data.Dataset.from_tensor_slices((left_contexts, left_contexts_len, right_contexts, right_contexts_len, labels))
    dataset = dataset.shuffle(buffer_size=len(labels))

    if batch_size==None:
        dataset = dataset.batch(batch_size=1)
    else:
        dataset = dataset.batch(batch_size=batch_size)

    dataset = dataset.map(lambda left,left_len,right,right_len,label: ({'left': {'x': left, 'len': left_len}, 'right': {'x': right, 'len': right_len}}, label))

    if batch_size!=None:
        dataset = dataset.repeat()

    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()

def tclstm_input_fn(features, labels, batch_size, embedding, max_seq_length, num_out_classes):
    embedding.set_embedding_matrix_variable()

    left_contexts =  tf.data.Dataset.from_generator(lambda: features['mappings']['left'], output_shapes=[None], output_types=tf.int32)
    targets = tf.data.Dataset.from_generator(lambda: features['mappings']['target'], output_shapes=[None], output_types=tf.int32)
    right_contexts = tf.data.Dataset.from_generator(lambda: features['mappings']['right'], output_shapes=[None], output_types=tf.int32)

    embedded_averaged_targets = targets.map(embed_target_and_average)
    embedded_left_contexts = left_contexts.map(embed_from_ids)
    embedded_right_contexts = right_contexts.map(lambda context: tf.reverse(context, axis=[0])).map(embed_from_ids)

    target_connected_left = tf.data.Dataset.zip((embedded_left_contexts, embedded_averaged_targets)).map(lambda left,target: tf.cond(tf.size(left)>0, lambda: tf.map_fn(lambda word: tf.concat([word, tf.squeeze(target)], axis=0), left),lambda: tf.zeros([max_seq_length, embedding.get_embedding_dim()*2])))
    target_connected_right = tf.data.Dataset.zip((embedded_right_contexts, embedded_averaged_targets)).map(lambda right,target: tf.cond(tf.size(right)>0, lambda: tf.map_fn(lambda word: tf.concat([word, tf.squeeze(target)], axis=0), right),lambda: tf.zeros([max_seq_length, embedding.get_embedding_dim()*2])))

    sparse_features_left = target_connected_left.apply(tf.contrib.data.dense_to_sparse_batch(batch_size=1, row_shape=[max_seq_length,embedding.get_embedding_dim()*2]))
    sparse_features_right = target_connected_right.apply(tf.contrib.data.dense_to_sparse_batch(batch_size=1, row_shape=[max_seq_length,embedding.get_embedding_dim()*2]))
    sparse_features_dict = tf.data.Dataset.zip(({'left': {'x' : sparse_features_left}, 'right': {'x' : sparse_features_right}})) 

    labels_dataset = tf.data.Dataset.from_tensor_slices([label+1 for label in labels])

    dataset = tf.data.Dataset.zip((sparse_features_dict, labels_dataset))

    if batch_size!=None:
        return dataset.apply(tf.contrib.data.shuffle_and_repeat(len(labels))).batch(batch_size=batch_size)
    else:
        return dataset.batch(batch_size=1)

def dual_lstm_model_fn(features, labels, mode, params):
    with tf.variable_scope('left_lstm'):
        features['left']['x'] = tf.contrib.layers.dense_to_sparse(features['left']['x'])
        input_layer, sequence_length = tf.contrib.feature_column.sequence_input_layer(
            features=features['left'],
            feature_columns=params['feature_columns']
        )

        sequence_length = tf.Print(input_=sequence_length, data=[sequence_length], message='Seq length: ')

        _, final_states_left = tf.nn.dynamic_rnn(
            cell=shared_lstm_cell_with_dropout(params),
            inputs=input_layer,
            sequence_length=sequence_length,
            dtype=tf.float32
        )

    with tf.variable_scope('right_lstm'):
        features['right']['x'] = tf.contrib.layers.dense_to_sparse(features['right']['x'])
        input_layer, sequence_length = tf.contrib.feature_column.sequence_input_layer(
            features=features['right'],
            feature_columns=params['feature_columns']
        )

        sequence_length = tf.Print(input_=sequence_length, data=[sequence_length], message='Seq length: ')

        _, final_states_right = tf.nn.dynamic_rnn(
            cell=shared_lstm_cell_with_dropout(params),
            inputs=input_layer,
            sequence_length=sequence_length,
            dtype=tf.float32
        )
    
    concatenated_final_states = tf.concat([final_states_left.h, final_states_right.h], axis=1)

    logits = tf.layers.dense(inputs=concatenated_final_states, units=params['n_out_classes'])
    predicted_classes = tf.argmax(logits, 1)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': predicted_classes[:, tf.newaxis],
            'probabilities': tf.nn.softmax(logits),
            'logits': logits
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    accuracy = tf.metrics.accuracy(labels=labels,predictions=predicted_classes, name='acc_op')
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

    logging_hook = tf.train.LoggingTensorHook({'loss': loss, 'accuracy': accuracy[1]}, every_n_iter=50)

    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op, training_hooks=[logging_hook])
