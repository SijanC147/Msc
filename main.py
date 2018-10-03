# import comet_ml
from tsaplay.datasets.Dataset import Dataset, DATASETS
from tsaplay.embeddings.Embedding import Embedding
from tsaplay.features.FeatureProvider import FeatureProvider
from tsaplay.experiments.Experiment import Experiment

from tsaplay.models.Lstm import Lstm
from tsaplay.models.TCLstm import TCLstm
from tsaplay.models.TDLstm import TDLstm
from tsaplay.models.LCRRot import LCRRot
from tsaplay.models.Ian import Ian
from tsaplay.models.MemNet import MemNet
from tsaplay.models.Ram import Ram

dataset = Dataset(*DATASETS.DONG)
# glv = Embedding("glove-wiki-gigaword-50")
glv = Embedding("glove-cc840-300")
model = LCRRot(params={"hidden_units": 5, "batch_size": 5})

feature_provider = FeatureProvider(datasets=[dataset], embedding=glv)

experiment = Experiment(feature_provider, model)
experiment.run(job="train+eval", steps=100)
# experiment.launch_tensorboard()
# experiment.export_model()
