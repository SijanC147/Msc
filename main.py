import tensorflow as tf
tf.logging.set_verbosity(tf.logging.INFO)
from tensorflow.python import debug as tf_debug  # pylint: disable=E0611

from experiments.Experiment import Experiment
from embeddings.GloVe import GloVe
from datasets.Dong2014 import Dong2014
from datasets.Xue2018 import Xue2018
from datasets.Wang2017 import Wang2017
from datasets.Rosenthal2015 import Rosenthal2015
from models.Tang2016a.LSTM import LSTM
from models.Tang2016a.TCLSTM import TCLSTM
from models.Tang2016a.TDLSTM import TDLSTM

experiment = Experiment(
    dataset=Rosenthal2015(),
    embedding=GloVe(alias='twitter', version='200'),
    model=LSTM(),
    # custom_tag='1',
    # continue_training=True
)
experiment.run(mode='train+eval', steps=300, open_tensorboard=False)

# tdlstm_experiment = Experiment(
#     name='setting a baseline',
#     dataset=Dong2014(),
#     embedding=glove,
#     model=TCLSTM()
# )
# tclstm_experiment = Experiment(
#     name='setting a baseline',
#     dataset=Dong2014(),
#     embedding=glove,
#     model=TCLSTM()
# )

# experiment.run(mode='train', steps=1, batch_size=1, debug=True, open_tensorboard=True)
# experiment.run(mode='train', steps=100, batch_size=1, debug=True, train_hooks=[tf_debug.TensorBoardDebugHook("127.0.0.1:6064")])
# lstm_experiment.run(mode='train', steps=1500, batch_size=25)
# tclstm_experiment.run(mode='train', steps=5000)
# tdlstm_experiment.run(mode='train', steps=5000)
