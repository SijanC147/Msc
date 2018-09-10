import tensorflow as tf
from tensorflow.python import debug as tf_debug  # pylint: disable=E0611
from experiments.Experiment import Experiment
from datasets.Dataset import Dataset, DATASETS
from embeddings.Embedding import Embedding, EMBEDDINGS
from models.Tang2016a.LSTM import LSTM
from models.Tang2016a.TDLSTM import TDLSTM
from models.Tang2016a.TCLSTM import TCLSTM

tf.logging.set_verbosity(tf.logging.INFO)


embedding = Embedding(path=EMBEDDINGS.DEBUG)

dataset = Dataset(
    path=DATASETS.DEBUG["PATH"],
    parser=DATASETS.DEBUG["PARSER"],
    embedding=embedding,
)
model = LSTM(run_config=tf.estimator.RunConfig(tf_random_seed=1234))

experiment = Experiment(dataset=dataset, embedding=embedding, model=model)

experiment.run(job="train+eval", steps=100)

