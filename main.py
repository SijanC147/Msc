import tensorflow as tf
import numpy as np
from tensorflow.python import debug as tf_debug  # pylint: disable=E0611
from tsaplay.datasets.Dataset import Dataset, DATASETS
from tsaplay.embeddings.Embedding import Embedding, EMBEDDINGS
from tsaplay.experiments.Experiment import Experiment

# from tsaplay.models.Tang2016a.Lstm import Lstm
# from tsaplay.models.Tang2016a.TcLstm import TcLstm
# from tsaplay.models.Tang2016a.TdLstm import TdLstm
# from tsaplay.models.Zheng2018.LcrRot import LcrRot
# from tsaplay.models.Tang2016b.MemNet import MemNet
from tsaplay.models.Chen2017.RecurrentAttentionNetwork import (
    RecurrentAttentionNetwork
)
from tsaplay.models.Ma2017.InteractiveAttentionNetwork import (
    InteractiveAttentionNetwork
)

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

# embedding = Embedding(path=EMBEDDINGS.GLOVE_TWITTER_100D)
# embedding.export_vocabulary_file()


# experiment = Experiment(
#     dataset=Dataset(
#         path=DATASETS.DEBUG_PATH,
#         parser=DATASETS.DEBUG_PARSER,
#         embedding=Embedding(path=EMBEDDINGS.DEBUG),
#         filter_embedding=False,
#     ),
#     model=RecurrentAttentionNetwork(),
#     run_config=tf.estimator.RunConfig(tf_random_seed=1234),
# )
# experiment.run(job="train+eval", steps=5)
# # experiment.export_model(overwrite=True)
experiment = Experiment(
    dataset=Dataset(
        path=DATASETS.DONG2014_PATH,
        parser=DATASETS.DONG2014_PARSER,
        embedding=Embedding(source="glove-twitter-25"),
        filter_embedding=False,
    ),
    model=RecurrentAttentionNetwork(),
    # contd_tag="gold",
    # run_config=tf.estimator.RunConfig(tf_random_seed=1234),
)
experiment.run(job="train+eval", steps=300)
# # experiment.export_model(overwrite=True)
# experiment.export_model(overwrite=True, restart_tfserve=True)
