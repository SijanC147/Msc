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

dong = Dataset(path=DATASETS.DONG2014_PATH, parser=DATASETS.DONG2014_PARSER)
nakov = Dataset(path=DATASETS.NAKOV2016_PATH, parser=DATASETS.NAKOV2016_PARSER)
rosenthal = Dataset(
    path=DATASETS.ROSENTHAL2015_PATH, parser=DATASETS.ROSENTHAL2015_PARSER
)
saeidi = Dataset(
    path=DATASETS.SAEIDI2016_PATH, parser=DATASETS.SAEIDI2016_PARSER
)
wang = Dataset(path=DATASETS.WANG2017_PATH, parser=DATASETS.WANG2017_PARSER)
xue = Dataset(path=DATASETS.XUE2018_PATH, parser=DATASETS.XUE2018_PARSER)
model = Lstm()

experiment = Experiment(
    dataset=dong,
    embedding=Embedding(path=EMBEDDINGS.GLOVE_TWITTER_25D),
    model=model,
    run_config=run_config,
)
experiment.run(job="train+eval", steps=100)
experiment = Experiment(
    dataset=nakov,
    embedding=Embedding(path=EMBEDDINGS.GLOVE_TWITTER_25D),
    model=model,
    run_config=run_config,
)
experiment.run(job="train+eval", steps=100)
experiment = Experiment(
    dataset=rosenthal,
    embedding=Embedding(path=EMBEDDINGS.GLOVE_TWITTER_25D),
    model=model,
    run_config=run_config,
)
experiment.run(job="train+eval", steps=100)
experiment = Experiment(
    dataset=saeidi,
    embedding=Embedding(path=EMBEDDINGS.GLOVE_TWITTER_25D),
    model=model,
    run_config=run_config,
)
experiment.run(job="train+eval", steps=100)
experiment = Experiment(
    dataset=wang,
    embedding=Embedding(path=EMBEDDINGS.GLOVE_TWITTER_25D),
    model=model,
    run_config=run_config,
)
experiment.run(job="train+eval", steps=100)
experiment = Experiment(
    dataset=xue,
    embedding=Embedding(path=EMBEDDINGS.GLOVE_TWITTER_25D),
    model=model,
    run_config=run_config,
)
experiment.run(job="train+eval", steps=100)
