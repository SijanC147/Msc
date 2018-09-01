from tensorflow.python import debug as tf_debug

from experiments.Experiment import Experiment
from datasets.Dong2014 import Dong2014
from embeddings.GloVe import GloVe
from models.Tang2016a.LSTM import LSTM
from models.Tang2016a.TCLSTM import TCLSTM
from models.Tang2016a.TDLSTM import TDLSTM


glove = GloVe(alias='twitter', version='debug')

lstm_experiment = Experiment(
    name='checking dataset changes 2',
    dataset=Dong2014(),
    embedding=glove,
    model=LSTM()
)
# tdlstm_experiment = Experiment(
#     name='checking dataset changes',
#     dataset=Dong2014(),
#     embedding=glove,
#     model=TCLSTM()
# )
# tclstm_experiment = Experiment(
#     name='checking dataset changes',
#     dataset=Dong2014(),
#     embedding=glove,
#     model=TCLSTM()
# )

# lstm_experiment.run(mode='train', steps=5, batch_size=25, hooks=[tf_debug.TensorBoardDebugHook("127.0.0.1:6064")])
lstm_experiment.run(mode='train', steps=1000, batch_size=1, debug=True)
# tclstm_experiment.run(mode='train', steps=5000)
# tdlstm_experiment.run(mode='train', steps=5000)