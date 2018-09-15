import tensorflow as tf
import numpy as np
from tensorflow.python import debug as tf_debug  # pylint: disable=E0611
from tsaplay.datasets.Dataset import Dataset, DATASETS
from tsaplay.embeddings.Embedding import Embedding, EMBEDDINGS
from tsaplay.models.Tang2016a.Lstm import Lstm
from tsaplay.models.Tang2016a.TdLstm import TdLstm
from tsaplay.models.Tang2016a.TcLstm import TcLstm
from tsaplay.models.Zheng2018.LcrRot import LcrRot
from tsaplay.models.Ma2017.InteractiveAttentionNetwork import (
    InteractiveAttentionNetwork
)
from tsaplay.experiments.Experiment import Experiment

tf.logging.set_verbosity(tf.logging.INFO)


lstm = Lstm()
tdlstm = TdLstm()
tclstm = TcLstm()
lcrrot = LcrRot()
lcrrot = LcrRot()
ian = InteractiveAttentionNetwork()

experiment = Experiment(
    dataset=Dataset(
        path=DATASETS.XUE2018_PATH,
        parser=DATASETS.XUE2018_PARSER,
        embedding=Embedding(
            path=EMBEDDINGS.GLOVE_TWITTER_25D,
            oov=lambda size: np.random.uniform(low=0.1, high=0.1, size=size),
        ),
    ),
    model=ian,
    run_config=tf.estimator.RunConfig(tf_random_seed=1234),
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
