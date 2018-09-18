import tensorflow as tf
import numpy as np
from tensorflow.python import debug as tf_debug  # pylint: disable=E0611
from tsaplay.datasets.Dataset import Dataset, DATASETS
from tsaplay.embeddings.Embedding import Embedding, EMBEDDINGS
from tsaplay.models.Tang2016a.Lstm import Lstm
from tsaplay.models.Zheng2018.LcrRot import LcrRot
from tsaplay.experiments.Experiment import Experiment

tf.logging.set_verbosity(tf.logging.INFO)

# experiment = Experiment(
#     dataset=Dataset(
#         path=DATASETS.DEBUG_PATH,
#         parser=DATASETS.DEBUG_PARSER,
#         embedding=Embedding(path=EMBEDDINGS.DEBUG),
#     ),
#     model=LcrRot(),
#     run_config=tf.estimator.RunConfig(tf_random_seed=1234),
# )
# experiment.run(job="train+eval", steps=10, start_tb=True)
# experiment.run(job="train", steps=200, hooks=[tf_debug.LocalCLIDebugHook()])
# experiment = Experiment(
#     dataset=Dataset(
#         path=DATASETS.DONG2014_PATH,
#         parser=DATASETS.DONG2014_PARSER,
#         embedding=Embedding(path=EMBEDDINGS.GLOVE_TWITTER_25D),
#     ),
#     model=LcrRot(),
#     run_config=tf.estimator.RunConfig(tf_random_seed=1234),
# )
# experiment.run(job="train+eval", steps=1000)
# experiment = Experiment(
#     dataset=Dataset(
#         path=DATASETS.NAKOV2016_PATH,
#         parser=DATASETS.NAKOV2016_PARSER,
#         embedding=Embedding(path=EMBEDDINGS.GLOVE_TWITTER_25D),
#     ),
#     model=LcrRot(),
#     run_config=tf.estimator.RunConfig(tf_random_seed=1234),
# )
# experiment.run(job="train+eval", steps=1000)
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
experiment = Experiment(
    dataset=Dataset(
        path=DATASETS.WANG2017_PATH,
        parser=DATASETS.WANG2017_PARSER,
        embedding=Embedding(path=EMBEDDINGS.GLOVE_TWITTER_25D),
    ),
    model=LcrRot(),
    run_config=tf.estimator.RunConfig(tf_random_seed=1234),
)
experiment.run(job="train+eval", steps=1000)
experiment = Experiment(
    dataset=Dataset(
        path=DATASETS.XUE2018_RESTAURANTS_PATH,
        parser=DATASETS.XUE2018_PARSER,
        embedding=Embedding(path=EMBEDDINGS.GLOVE_TWITTER_25D),
    ),
    model=LcrRot(),
    run_config=tf.estimator.RunConfig(tf_random_seed=1234),
)
experiment.run(job="train+eval", steps=1000)
experiment = Experiment(
    dataset=Dataset(
        path=DATASETS.XUE2018_LAPTOPS_PATH,
        parser=DATASETS.XUE2018_PARSER,
        embedding=Embedding(path=EMBEDDINGS.GLOVE_TWITTER_25D),
    ),
    model=LcrRot(),
    run_config=tf.estimator.RunConfig(tf_random_seed=1234),
)
experiment.run(job="train+eval", steps=1000)
experiment = Experiment(
    dataset=Dataset(
        path=DATASETS.ROSENTHAL2015_PATH,
        parser=DATASETS.ROSENTHAL2015_PARSER,
        embedding=Embedding(path=EMBEDDINGS.GLOVE_TWITTER_25D),
    ),
    model=LcrRot(),
    run_config=tf.estimator.RunConfig(tf_random_seed=1234),
)
experiment.run(job="train+eval", steps=1000)
