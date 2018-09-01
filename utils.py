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
        return tokens_list
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
