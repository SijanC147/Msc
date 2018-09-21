import tensorflow as tf
import numpy as np
from tensorflow.python import debug as tf_debug  # pylint: disable=E0611
from tsaplay.datasets.Dataset import Dataset, DATASETS
from tsaplay.embeddings.Embedding import Embedding, EMBEDDINGS
from tsaplay.models.Tang2016a.Lstm import Lstm
from tsaplay.models.Zheng2018.LcrRot import LcrRot
from tsaplay.models.Ma2017.InteractiveAttentionNetwork import (
    InteractiveAttentionNetwork
)
from tsaplay.models.Tang2016b.MemNet import MemNet
from tsaplay.experiments.Experiment import Experiment

tf.logging.set_verbosity(tf.logging.INFO)

debug_hook = tf_debug.LocalCLIDebugHook()

debug_params = {
    "batch_size": 5,
    "max_seq_length": 20,
    "n_out_classes": 3,
    "learning_rate": 0.1,
    "l2_weight": 1e-5,
    "momentum": 0.9,
    "keep_prob": 0.5,
    "hidden_units": 5,
    "initializer": tf.initializers.random_uniform(minval=-0.1, maxval=0.1),
}

experiment = Experiment(
    dataset=Dataset(
        path=DATASETS.DEBUG_PATH,
        parser=DATASETS.DEBUG_PARSER,
        embedding=Embedding(path=EMBEDDINGS.DEBUG),
    ),
    model=Lstm(),
    contd_tag="gold_debug",
    # run_config=tf.estimator.RunConfig(tf_random_seed=1234),
)
# experiment.run(job="train+eval", steps=1, start_tb=True)
# # experiment.run(job="train", steps=200, hooks=[debug_hook])
# experiment = Experiment(
#     dataset=Dataset(
#         path=DATASETS.DONG2014_PATH,
#         parser=DATASETS.DONG2014_PARSER,
#         embedding=Embedding(path=EMBEDDINGS.GLOVE_TWITTER_25D),
#     ),
#     model=Lstm(),
#     contd_tag="gold",
#     # run_config=tf.estimator.RunConfig(tf_random_seed=1234),
# )
experiment.run(job="train+eval", steps=1)
experiment.export_model()
# experiment = Experiment(
#     dataset=Dataset(
#         path=DATASETS.NAKOV2016_PATH,
#         parser=DATASETS.NAKOV2016_PARSER,
#         embedding=Embedding(path=EMBEDDINGS.GLOVE_TWITTER_25D),
#     ),
#     model=LcrRot(),
#     contd_tag="gold_100D"
#     # run_config=tf.estimator.RunConfig(tf_random_seed=1234),
# )
# experiment.run(job="train+eval", steps=500, start_tb=True)
# experiment = Experiment(
#     dataset=Dataset(
#         path=DATASETS.SAEIDI2016_PATH,
#         parser=DATASETS.SAEIDI2016_PARSER,
#         embedding=Embedding(path=EMBEDDINGS.GLOVE_TWITTER_25D),
#     ),
#     model=LcrRot(),
#     run_config=tf.estimator.RunConfig(tf_random_seed=1234),
# )
# experiment.run(job="train+eval", steps=1000)
# experiment = Experiment(
#     dataset=Dataset(
#         path=DATASETS.WANG2017_PATH,
#         parser=DATASETS.WANG2017_PARSER,
#         embedding=Embedding(path=EMBEDDINGS.GLOVE_TWITTER_25D),
#     ),
#     model=LcrRot(),
#     run_config=tf.estimator.RunConfig(tf_random_seed=1234),
# )
# experiment.run(job="train+eval", steps=1000)
# experiment = Experiment(
#     dataset=Dataset(
#         path=DATASETS.XUE2018_RESTAURANTS_PATH,
#         parser=DATASETS.XUE2018_PARSER,
#         embedding=Embedding(path=EMBEDDINGS.GLOVE_TWITTER_100D),
#     ),
#     model=LcrRot(),
#     # run_config=tf.estimator.RunConfig(tf_random_seed=1234),
# )
# experiment.run(job="train+eval", steps=500)
# experiment = Experiment(
#     dataset=Dataset(
#         path=DATASETS.XUE2018_LAPTOPS_PATH,
#         parser=DATASETS.XUE2018_PARSER,
#         embedding=Embedding(path=EMBEDDINGS.GLOVE_TWITTER_100D),
#     ),
#     model=LcrRot(),
#     # run_config=tf.estimator.RunConfig(tf_random_seed=1234),
# )
# experiment.run(job="train+eval", steps=500, start_tb=True)
# experiment = Experiment(
#     dataset=Dataset(
#         path=DATASETS.ROSENTHAL2015_PATH,
#         parser=DATASETS.ROSENTHAL2015_PARSER,
#         embedding=Embedding(path=EMBEDDINGS.GLOVE_TWITTER_25D),
#     ),
#     model=LcrRot(),
#     run_config=tf.estimator.RunConfig(tf_random_seed=1234),
# )
# experiment.run(job="train+eval", steps=1000)
