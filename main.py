import tensorflow as tf
import numpy as np
from tensorflow.python import debug as tf_debug  # pylint: disable=E0611
from tsaplay.datasets.Dataset import Dataset, DATASETS
from tsaplay.embeddings.Embedding import Embedding, EMBEDDINGS
from tsaplay.models.Tang2016a.Lstm import Lstm
from tsaplay.models.Zheng2018.LcrRot import LcrRot
from tsaplay.experiments.Experiment import Experiment

tf.logging.set_verbosity(tf.logging.INFO)

experiment = Experiment(
    dataset=Dataset(
        path=DATASETS.DONG2014_PATH,
        parser=DATASETS.DONG2014_PARSER,
        embedding=Embedding(path=EMBEDDINGS.GLOVE_TWITTER_25D),
    ),
    model=Lstm(),
    run_config=tf.estimator.RunConfig(tf_random_seed=1234),
)
experiment.run(job="train+eval", steps=500, start_tb=True)
# experiment.run(job="train", steps=200, hooks=[tf_debug.LocalCLIDebugHook()])
