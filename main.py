import tensorflow as tf
from tensorflow.python import debug as tf_debug  # pylint: disable=E0611
from tsaplay.datasets.Dataset import Dataset, DATASETS
from tsaplay.embeddings.Embedding import Embedding, EMBEDDINGS
from tsaplay.models.Tang2016a.Lstm import Lstm
from tsaplay.models.Tang2016a.TdLstm import TdLstm
from tsaplay.models.Tang2016a.TcLstm import TcLstm
from tsaplay.models.Zheng2018.LcrRot import LcrRot
from tsaplay.experiments.Experiment import Experiment

tf.logging.set_verbosity(tf.logging.INFO)


# lstm = Lstm(run_config=tf.estimator.RunConfig(tf_random_seed=1234))
# tdlstm = TdLstm(run_config=tf.estimator.RunConfig(tf_random_seed=1234))
tclstm = TcLstm(run_config=tf.estimator.RunConfig(tf_random_seed=1234))
# lcrrot = LcrRot(run_config=tf.estimator.RunConfig(tf_random_seed=1234))
lcrrot = LcrRot(run_config=tf.estimator.RunConfig())

experiment = Experiment(
    dataset=Dataset(
        path=DATASETS.DONG2014_PATH,
        parser=DATASETS.DONG2014_PARSER,
        embedding=Embedding(path=EMBEDDINGS.GLOVE_TWITTER_25D),
    ),
    model=lcrrot,
)
experiment.run(job="train+eval", steps=1000)

# experiment = Experiment(
#     dataset=Dataset(
#         path=DATASETS.XUE2018_PATH,
#         parser=DATASETS.XUE2018_PARSER,
#         embedding=Embedding(path=EMBEDDINGS.GLOVE_TWITTER_25D),
#     ),
#     model=lcrrot,
# )
# experiment.run(job="train", steps=200, hooks=[tf_debug.LocalCLIDebugHook()])
