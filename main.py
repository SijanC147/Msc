import tensorflow as tf
from tensorflow.python import debug as tf_debug  # pylint: disable=E0611
from tsaplay.datasets.Dataset import Dataset, DATASETS
from tsaplay.embeddings.Embedding import Embedding, EMBEDDINGS
from tsaplay.models.Tang2016a.Lstm import Lstm
from tsaplay.models.Tang2016a.TdLstm import TdLstm
from tsaplay.models.Tang2016a.TcLstm import TcLstm
from tsaplay.experiments.Experiment import Experiment

tf.logging.set_verbosity(tf.logging.INFO)


model = Lstm(run_config=tf.estimator.RunConfig(tf_random_seed=1234))

# experiment = Experiment(
#     dataset=Dataset(
#         path=DATASETS.DONG2014_PATH,
#         parser=DATASETS.DONG2014_PARSER,
#         embedding=Embedding(path=EMBEDDINGS.GLOVE_TWITTER_25D),
#     ),
#     model=model,
# )
# experiment.run(job="train+eval", steps=100)
# experiment = Experiment(
#     dataset=Dataset(
#         path=DATASETS.NAKOV2016_PATH,
#         parser=DATASETS.NAKOV2016_PARSER,
#         embedding=Embedding(path=EMBEDDINGS.GLOVE_TWITTER_25D),
#     ),
#     model=model,
# )
# experiment.run(job="train+eval", steps=100)
# experiment = Experiment(
#     dataset=Dataset(
#         path=DATASETS.SAEIDI2016_PATH,
#         parser=DATASETS.SAEIDI2016_PARSER,
#         embedding=Embedding(path=EMBEDDINGS.GLOVE_TWITTER_25D),
#     ),
#     model=model,
# )
# experiment.run(job="train+eval", steps=100)
# experiment = Experiment(
#     dataset=Dataset(
#         path=DATASETS.SAEIDI2016_PATH,
#         parser=DATASETS.SAEIDI2016_PARSER,
#         embedding=Embedding(path=EMBEDDINGS.GLOVE_TWITTER_25D),
#     ),
#     model=model,
# )
# experiment.run(job="train+eval", steps=100)
# experiment = Experiment(
#     dataset=Dataset(
#         path=DATASETS.WANG2017_PATH,
#         parser=DATASETS.WANG2017_PARSER,
#         embedding=Embedding(path=EMBEDDINGS.GLOVE_TWITTER_25D),
#     ),
#     model=model,
# )
# experiment.run(job="train+eval", steps=100)
# experiment = Experiment(
#     dataset=Dataset(
#         path=DATASETS.XUE2018_PATH,
#         parser=DATASETS.XUE2018_PARSER,
#         embedding=Embedding(path=EMBEDDINGS.GLOVE_TWITTER_25D),
#     ),
#     model=model,
# )
# experiment.run(job="train+eval", steps=100)
