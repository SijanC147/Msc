import tensorflow as tf
from tensorflow.python import debug as tf_debug  # pylint: disable=E0611
from tsaplay.datasets.Dataset import Dataset, DATASETS
from tsaplay.embeddings.Embedding import Embedding, EMBEDDINGS
from tsaplay.models.Tang2016a.Lstm import Lstm
from tsaplay.models.Tang2016a.TdLstm import TdLstm
from tsaplay.models.Tang2016a.TcLstm import TcLstm
from tsaplay.experiments.Experiment import Experiment

tf.logging.set_verbosity(tf.logging.INFO)


# lstm = Lstm(run_config=tf.estimator.RunConfig(tf_random_seed=1234))
tdlstm = TdLstm(run_config=tf.estimator.RunConfig(tf_random_seed=1234))
# tclstm = TcLstm(run_config=tf.estimator.RunConfig(tf_random_seed=1234))

experiment = Experiment(
    dataset=Dataset(
        path=DATASETS.NAKOV2016_PATH,
        parser=DATASETS.NAKOV2016_PARSER,
        embedding=Embedding(path=EMBEDDINGS.GLOVE_TWITTER_25D),
    ),
    model=tdlstm,
)
experiment.run(job="train+eval", steps=200, start_tb=True, tb_port=6008)
