from tensorflow.python import debug as tf_debug  # pylint: disable=E0611
from experiments.Experiment import Experiment
from embeddings.GloVe import GloVe
from datasets.Dong2014 import Dong2014
from datasets.Xue2018 import Xue2018
from datasets.Wang2017 import Wang2017
from datasets.Rosenthal2015 import Rosenthal2015
from models.Tang2016a.LSTM import LSTM

# from models.Tang2016a.TDLSTM import TDLSTM
# from models.Tang2016a.TCLSTM import TCLSTM

import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

experiment = Experiment(
    dataset=Dong2014(),
    embedding=GloVe(alias="twitter", version="25"),
    model=LSTM(),
    seed=1234,
)
experiment.run(job="train+eval", steps=100)

# experiment.run(job="train", steps=1000, open_tensorboard=False)
# experiment.run(job="train", steps=10, debug=True, open_tensorboard=True)
# experiment.run(
#     job="train",
#     steps=100,
#     debug=True,
#     train_hooks=[tf_debug.TensorBoardDebugHook("127.0.0.1:6064")],
# )
