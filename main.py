import tensorflow as tf
from tensorflow.python import debug as tf_debug  # pylint: disable=E0611
from datasets.Dataset import Dataset, DATASETS
from embeddings.Embedding import Embedding, EMBEDDINGS
from models.Tang2016a.Lstm import Lstm
from models.Tang2016a.TdLstm import TdLstm
from models.Tang2016a.TcLstm import TcLstm
from experiments.Experiment import Experiment

tf.logging.set_verbosity(tf.logging.INFO)

run_config = tf.estimator.RunConfig(tf_random_seed=1234)

embedding = Embedding(path=EMBEDDINGS.GLOVE_TWITTER_25D)
dataset = Dataset(path=DATASETS.DONG2014_PATH, parser=DATASETS.DONG2014_PARSER)
model = Lstm()

experiment = Experiment(
    dataset=dataset, embedding=embedding, model=model, run_config=run_config
)
experiment.run(job="train+eval", steps=100)
