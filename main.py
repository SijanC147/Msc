import tensorflow as tf

import time
import numpy as np

from datasets.Dong2014 import Dong2014
from embeddings.GloVe import GloVe
from models.Tang2016a.LSTM import LSTM

glove = GloVe(alias='42B', version='300')
dong = Dong2014(embedding=glove, rebuild_corpus=False)

lstm = LSTM('new_experiment', dong)

lstm.train(
    steps=10,
    batch_size=100
)
