import tensorflow as tf
import numpy as np
from tensorflow.python import debug as tf_debug  # pylint: disable=E0611
from tsaplay.datasets.Dataset import Dataset, DATASETS
from tsaplay.embeddings.Embedding import Embedding, EMBEDDINGS
from tsaplay.models.Ma2017.InteractiveAttentionNetwork import (
    InteractiveAttentionNetwork
)
from tsaplay.models.Tang2016a.Lstm import Lstm
from tsaplay.experiments.Experiment import Experiment

tf.logging.set_verbosity(tf.logging.INFO)

ian = InteractiveAttentionNetwork(
    run_config=tf.estimator.RunConfig(tf_random_seed=1234)
)
lstm = Lstm(run_config=tf.estimator.RunConfig(tf_random_seed=1234))
experiment = Experiment(
    dataset=Dataset(
        path=DATASETS.DEBUG_PATH,
        parser=DATASETS.DEBUG_PARSER,
        embedding=Embedding(
            path=EMBEDDINGS.DEBUG,
            oov=lambda size: np.random.uniform(low=0.1, high=0.1, size=size),
        ),
    ),
    model=ian,
)
experiment.run(job="train+eval", steps=400, start_tb=True)
# experiment.run(job="train", steps=200, hooks=[tf_debug.LocalCLIDebugHook()])
