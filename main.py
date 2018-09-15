import tensorflow as tf
import numpy as np
from tensorflow.python import debug as tf_debug  # pylint: disable=E0611
from tsaplay.datasets.Dataset import Dataset, DATASETS
from tsaplay.embeddings.Embedding import Embedding, EMBEDDINGS
from tsaplay.models.Ma2017.InteractiveAttentionNetwork import (
    InteractiveAttentionNetwork
)
from tsaplay.experiments.Experiment import Experiment

tf.logging.set_verbosity(tf.logging.INFO)

ian = InteractiveAttentionNetwork(
    run_config=tf.estimator.RunConfig(tf_random_seed=1234)
)
experiment = Experiment(
    dataset=Dataset(
        path=DATASETS.DONG2014_PATH,
        parser=DATASETS.DONG2014_PARSER,
        embedding=Embedding(
            path=EMBEDDINGS.GLOVE_TWITTER_25D,
            oov=lambda size: np.random.uniform(low=0.1, high=0.1, size=size),
        ),
    ),
    model=ian,
)
experiment.run(job="train+eval", steps=600, start_tb=True)
# experiment.run(job="train", steps=200, hooks=[tf_debug.LocalCLIDebugHook()])
