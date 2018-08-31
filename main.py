from experiments.Experiment import Experiment
from datasets.Dong2014 import Dong2014
from embeddings.GloVe import GloVe
from models.Tang2016a.LSTM import LSTM
from models.Tang2016a.TCLSTM import TCLSTM
from models.Tang2016a.TDLSTM import TDLSTM


glove = GloVe(alias='42B', version='300')
dong = Dong2014()
# lstm = LSTM()
# tclstm = TCLSTM()
tdlstm = TDLSTM()

# tc_experiment = Experiment(
#     name='trying this shit out',
#     dataset=dong,
#     embedding=glove,
#     model=tclstm
# )
td_experiment = Experiment(
    name='trying this shit out',
    dataset=dong,
    embedding=glove,
    model=tdlstm
)

# tc_experiment.run(
#     mode='train', 
#     steps=10, 
#     batch_size=10
# )
td_experiment.run(
    mode='train', 
    steps=10 
)