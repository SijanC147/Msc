import tensorflow as tf
from tensorflow.python import debug as tf_debug  # pylint: disable=E0611
from experiments.Experiment import Experiment
from datasets.Dataset import Dataset, DATASETS
from embeddings.Embedding import Embedding, EMBEDDINGS
from models.Tang2016a.Lstm import Lstm
from models.Tang2016a.TdLstm import TdLstm
from models.Tang2016a.TcLstm import TcLstm

tf.logging.set_verbosity(tf.logging.INFO)

run_config = tf.estimator.RunConfig(tf_random_seed=1234)

embedding = Embedding(path=EMBEDDINGS.DEBUG)
dataset = Dataset(path=DATASETS.DEBUG_PATH, parser=DATASETS.DEBUG_PARSER)
model = Lstm(run_config=run_config)

experiment = Experiment(dataset=dataset, embedding=embedding, model=model)
experiment.run(job="train+eval", steps=100)
