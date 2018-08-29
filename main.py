import tensorflow as tf
from tensorflow.python import debug as tf_debug

import time
import numpy as np

from datasets.Dong2014 import Dong2014
from embeddings.GloVe import GloVe
from models.Tang2016a.LSTM import LSTM

glove = GloVe(alias='twitter', version='200')

start = time.time()
dong = Dong2014(embedding=glove, rebuild_corpus=False)
end = time.time()
print('Dataset created in: ' + str(end-start) + ' seconds')

start = time.time()
features, labels = dong.get_mapped_features_and_labels(mode='train')
end = time.time()
print('Loaded dataset in: ' + str(end-start) + ' seconds')

start = time.time()
lstm = LSTM (
    feature_columns = [tf.contrib.feature_column.sequence_numeric_column(key='x')],
    embedding = glove,
    model_dir = 'new dataset parent class'
    )
end = time.time()
print('Created LSTM object in: ' + str(end-start) + ' seconds')

start = time.time()
classifier = tf.estimator.Estimator(model_fn=lstm.model_fn, params=lstm.params, model_dir=lstm.get_model_dir())
end = time.time()
print('Created classifier object in: ' + str(end-start) + ' seconds')

print('Starting training, hang tight')
start = time.time()
classifier.train(
    input_fn=lambda: lstm.train_input_fn(features, labels, 100), 
    steps=500,
    hooks=[])
    # hooks=[tf_debug.TensorBoardDebugHook("127.0.0.1:6064")])
end = time.time()
print('Completed training in: ' + str(end-start) + ' seconds')