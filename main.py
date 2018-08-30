from experiments.Experiment import Experiment
from datasets.Dong2014 import Dong2014
from embeddings.GloVe import GloVe
from models.Tang2016a.LSTM import LSTM

glove = GloVe(alias='42B', version='300')
dong = Dong2014()
lstm = LSTM()

exp = Experiment(
    name='trying this shit out',
    dataset=dong,
    embedding=glove,
    model=lstm
)

exp.run_experiment('train', 10, 10)

