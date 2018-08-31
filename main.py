from experiments.Experiment import Experiment
from datasets.Dong2014 import Dong2014
from embeddings.GloVe import GloVe
from models.Tang2016a.TCLSTM import TCLSTM

glove = GloVe(alias='42B', version='300')
dong = Dong2014()
tclstm = TCLSTM()

experiment = Experiment(
    name='trying this shit out',
    dataset=dong,
    embedding=glove,
    model=tclstm
)

experiment.run(
    mode='train', 
    steps=10, 
    batch_size=10
    )

