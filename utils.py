import spacy
import nltk
import random
import math
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

def change_features_labels_distribution(features, labels, positive, neutral, negative):
    counts = [len([l for l in labels if l==1]), len([l for l in labels if l==0]), len([l for l in labels if l==-1])]
    target_dists = [positive, neutral, negative]
    target_counts = [0,0,0]

    smallest_count_indices = [i for i,x in enumerate(counts) if x==min(counts)]
    if len(smallest_count_indices)!=1:
        smallest_count_index = target_dists.index(max([target_dists[i] for i in smallest_count_indices]))
    else:
        smallest_count_index = smallest_count_indices[0]

    target_counts[smallest_count_index] = counts[smallest_count_index]
    new_total = math.floor(counts[smallest_count_index]/target_dists[smallest_count_index])
    counts[smallest_count_index] = float('inf') 
    target_dists[smallest_count_index] = float('inf') 

    smallest_count_indices = [i for i,x in enumerate(counts) if x==min(counts)]
    if len(smallest_count_indices)!=1:
        smallest_count_index = target_dists.index(max([target_dists[i] for i in smallest_count_indices]))
    else:
        smallest_count_index = smallest_count_indices[0]
    smallest_count_index = counts.index(min(counts))
    target_count = int(new_total*target_dists[smallest_count_index])
    if(target_count>counts[smallest_count_index]):
        old_total = new_total
        new_total = math.floor(counts[smallest_count_index]/target_dists[smallest_count_index])
        target_counts = [math.floor(t*(new_total/old_total)) for t in target_counts]
        target_count = counts[smallest_count_index]
    target_counts[smallest_count_index] = target_count
    counts[smallest_count_index] = float('inf') 
    target_dists[smallest_count_index] = float('inf') 

    if(new_total-sum(target_counts)>min(counts)):
        old_total = new_total
        new_total = math.floor(min(counts)/min(target_dists))
        target_counts = [math.floor(t*(new_total/old_total)) for t in target_counts]
    target_counts[target_counts.index(0)] = new_total-sum(target_counts)
    
    postive_sample_indices = [i for i,x in enumerate(labels) if x==1]
    neutral_sample_indices = [i for i,x in enumerate(labels) if x==0]
    negative_sample_indices = [i for i,x in enumerate(labels) if x==-1]
    new_features = {
            'sentence' : ['']*new_total,
            'sentence_length': [0]*new_total,
            'target' : ['']*new_total,
            'mappings': {
                'left': [[]]*new_total,
                'target': [[]]*new_total,
                'right': [[]]*new_total
            },
        }
    new_labels = [None]*new_total

    for _ in range(target_counts[0]):
        random_index = random.choice(postive_sample_indices)
        random_position = random.randrange(0,new_total)
        while new_labels[random_position]!=None:
            random_position = random.randrange(0,new_total)
        new_features['sentence'][random_position]= features['sentence'][random_index]
        new_features['sentence_length'][random_position]= features['sentence_length'][random_index]
        new_features['target'][random_position]= features['target'][random_index]
        new_features['mappings']['left'][random_position]= features['mappings']['left'][random_index]
        new_features['mappings']['target'][random_position]= features['mappings']['target'][random_index]
        new_features['mappings']['right'][random_position]= features['mappings']['right'][random_index]
        new_labels[random_position]= labels[random_index]
        postive_sample_indices.remove(random_index)
    for _ in range(target_counts[1]):
        random_index = random.choice(neutral_sample_indices)
        random_position = random.randrange(0,new_total)
        while new_labels[random_position]!=None:
            random_position = random.randrange(0,new_total)
        new_features['sentence'][random_position]= features['sentence'][random_index]
        new_features['sentence_length'][random_position]= features['sentence_length'][random_index]
        new_features['target'][random_position]= features['target'][random_index]
        new_features['mappings']['left'][random_position]= features['mappings']['left'][random_index]
        new_features['mappings']['target'][random_position]= features['mappings']['target'][random_index]
        new_features['mappings']['right'][random_position]= features['mappings']['right'][random_index]
        new_labels[random_position]= labels[random_index]
        neutral_sample_indices.remove(random_index)
    for _ in range(target_counts[2]):
        random_index = random.choice(negative_sample_indices)
        random_position = random.randrange(0,new_total)
        while new_labels[random_position]!=None:
            random_position = random.randrange(0,new_total)
        new_features['sentence'][random_position]= features['sentence'][random_index]
        new_features['sentence_length'][random_position]= features['sentence_length'][random_index]
        new_features['target'][random_position]= features['target'][random_index]
        new_features['mappings']['left'][random_position]= features['mappings']['left'][random_index]
        new_features['mappings']['target'][random_position]= features['mappings']['target'][random_index]
        new_features['mappings']['right'][random_position]= features['mappings']['right'][random_index]
        new_labels[random_position]= labels[random_index]
        negative_sample_indices.remove(random_index)
    
    return new_features, new_labels


# def change_features_labels_distribution(features, labels, positive, neutral, negative):
#     counts = [len([l for l in labels if l==1]), len([l for l in labels if l==0]), len([l for l in labels if l==-1])]
#     target_dists = [positive, neutral, negative]
#     target_counts = [0,0,0]

#     highest_target_indices = [i for i,x in enumerate(target_dists) if x==max(target_dists)]
#     if len(highest_target_indices)!=1:
#         highest_target_index = counts.index(min([counts[i] for i in highest_target_indices]))
#     else:
#         highest_target_index = highest_target_indices[0]

#     target_counts[highest_target_index] = counts[highest_target_index]
#     new_total = math.floor(counts[highest_target_index]/target_dists[highest_target_index])
#     counts[highest_target_index] = -1
#     target_dists[highest_target_index] = -1

#     if len(set([dist for dist in target_dists if dist>=0]))==1:
#         target_counts[target_counts.index(0)] = int((new_total-sum(target_counts))/2)
#         target_counts[target_counts.index(0)] = new_total-sum(target_counts)
#     else:
#         next_highest_target_index = target_dists.index(max(target_dists))
#         target_counts[next_highest_target_index] = int(new_total*target_dists[next_highest_target_index])
#         target_counts[target_counts.index(0)] = new_total-sum(target_counts)
    
#     postive_sample_indices = [i for i,x in enumerate(labels) if x==1]
#     neutral_sample_indices = [i for i,x in enumerate(labels) if x==0]
#     negative_sample_indices = [i for i,x in enumerate(labels) if x==-1]
#     new_features = {
#             'sentence' : ['']*new_total,
#             'sentence_length': [0]*new_total,
#             'target' : ['']*new_total,
#             'mappings': {
#                 'left': [[]]*new_total,
#                 'target': [[]]*new_total,
#                 'right': [[]]*new_total
#             },
#         }
#     new_labels = [None]*new_total

#     for _ in range(target_counts[0]):
#         random_index = random.choice(postive_sample_indices)
#         random_position = random.randrange(0,new_total)
#         while new_labels[random_position]!=None:
#             random_position = random.randrange(0,new_total)
#         new_features['sentence'][random_position]= features['sentence'][random_index]
#         new_features['sentence_length'][random_position]= features['sentence_length'][random_index]
#         new_features['target'][random_position]= features['target'][random_index]
#         new_features['mappings']['left'][random_position]= features['mappings']['left'][random_index]
#         new_features['mappings']['target'][random_position]= features['mappings']['target'][random_index]
#         new_features['mappings']['right'][random_position]= features['mappings']['right'][random_index]
#         new_labels[random_position]= labels[random_index]
#         postive_sample_indices.remove(random_index)
#     for _ in range(target_counts[1]):
#         random_index = random.choice(neutral_sample_indices)
#         random_position = random.randrange(0,new_total)
#         while new_labels[random_position]!=None:
#             random_position = random.randrange(0,new_total)
#         new_features['sentence'][random_position]= features['sentence'][random_index]
#         new_features['sentence_length'][random_position]= features['sentence_length'][random_index]
#         new_features['target'][random_position]= features['target'][random_index]
#         new_features['mappings']['left'][random_position]= features['mappings']['left'][random_index]
#         new_features['mappings']['target'][random_position]= features['mappings']['target'][random_index]
#         new_features['mappings']['right'][random_position]= features['mappings']['right'][random_index]
#         new_labels[random_position]= labels[random_index]
#         neutral_sample_indices.remove(random_index)
#     for _ in range(target_counts[2]):
#         random_index = random.choice(negative_sample_indices)
#         random_position = random.randrange(0,new_total)
#         while new_labels[random_position]!=None:
#             random_position = random.randrange(0,new_total)
#         new_features['sentence'][random_position]= features['sentence'][random_index]
#         new_features['sentence_length'][random_position]= features['sentence_length'][random_index]
#         new_features['target'][random_position]= features['target'][random_index]
#         new_features['mappings']['left'][random_position]= features['mappings']['left'][random_index]
#         new_features['mappings']['target'][random_position]= features['mappings']['target'][random_index]
#         new_features['mappings']['right'][random_position]= features['mappings']['right'][random_index]
#         new_labels[random_position]= labels[random_index]
#         negative_sample_indices.remove(random_index)
    
#     return new_features, new_labels

def get_statistics_on_features_labels(features, labels):
    return {
        'num_samples': len(labels),
        'positive':{
            'count': len([label for label in labels if label==1]),
            'percent': round((len([label for label in labels if label==1])/len(labels))*100, 2)
        } ,
        'neutral': {
            'count': len([label for label in labels if label==0]),
            'percent': round((len([label for label in labels if label==0])/len(labels))*100, 2)
        } ,
        'negative': {
            'count': len([label for label in labels if label==-1]),
            'percent': round((len([label for label in labels if label==-1])/len(labels))*100, 2)
        },
        'mean_sen_length': round(np.mean(features['sentence_length']),2)}