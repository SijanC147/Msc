import spacy
import nltk
import numpy as np
import tensorflow as tf

def tokenize_phrase(phrase, backend='spacy'):
    if backend=='nltk':
        return(nltk.word_tokenize(phrase))
    elif backend=='spacy':
        tokens_list = []
        nlp = spacy.load('en')
        tokens = nlp(str(phrase))
        tokens_list = list(filter(keep_token, tokens))
        return [token.text for token in tokens_list]
    elif backend=='vanilla':
        return(phrase.split())

def get_embedding_matrix_variable():
    with tf.variable_scope('shared', reuse=tf.AUTO_REUSE):
        embedding_matrix = tf.get_variable('embedding_matrix', trainable=False)
    return embedding_matrix

def embed_target_and_average(target):
    embedding_matrix = get_embedding_matrix_variable()
    target_embedding = tf.cond(tf.shape(target)[0]>1, lambda: tf.reduce_mean(tf.nn.embedding_lookup(embedding_matrix, target), axis=0, keepdims=True),lambda: tf.nn.embedding_lookup(embedding_matrix, target))
    return target_embedding

def embed_and_concat(left, target, right):
    embedding_matrix = get_embedding_matrix_variable() 
    left_embedded = tf.nn.embedding_lookup(embedding_matrix, left)
    target_embedding = tf.nn.embedding_lookup(embedding_matrix, target)
    right_embedding = tf.nn.embedding_lookup(embedding_matrix, right)

    return tf.concat([left_embedded, target_embedding, right_embedding], axis=0)

def embed_from_ids(ids_tensor):
    embedding_matrix = get_embedding_matrix_variable()
    embedding = tf.nn.embedding_lookup(embedding_matrix, ids_tensor)
    return embedding

def keep_token(token):
    if token.like_url:
        return False
    if token.like_email:
        return False
    if token.text in ['\uFE0F']:
        return False 
    return True

def random_input_fn(features, labels, batch_size, embedding, max_seq_length, num_out_classes):
    left_contexts =  tf.data.Dataset.from_generator(lambda: features['mappings']['left'], output_shapes=[None], output_types=tf.int32)
    targets = tf.data.Dataset.from_generator(lambda: features['mappings']['target'], output_shapes=[None], output_types=tf.int32)
    right_contexts = tf.data.Dataset.from_generator(lambda: features['mappings']['right'], output_shapes=[None], output_types=tf.int32)

    zipped_features = tf.data.Dataset.zip((left_contexts, targets, right_contexts))
    embedded_features = zipped_features.map(lambda l,t,r: tf.random_normal(shape=tf.shape([tf.concat([l,t,r], axis=0)])))
    sparse_features = embedded_features.apply(tf.contrib.data.dense_to_sparse_batch(batch_size=1, row_shape=[max_seq_length,embedding.get_embedding_dim()]))
    sparse_features_dict = tf.data.Dataset.zip(({'x' : sparse_features})) 

    labels_dataset = tf.data.Dataset.from_tensor_slices([label+1 for label in labels])

    dataset = tf.data.Dataset.zip((sparse_features_dict, labels_dataset))

    if batch_size!=None:
        return dataset.shuffle(len(features['sentence'])).repeat().batch(batch_size=batch_size)
    else:
        return dataset.shuffle(len(features['sentence'])).batch(batch_size=1)