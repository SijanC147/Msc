import comet_ml
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
glv = Embedding("glove-cc42-300")
model = LCRRot()

feature_provider = FeatureProvider(datasets=[dataset], embedding=glv)

experiment = Experiment(feature_provider, model, contd_tag="common-crawl-42")
experiment.run(job="train+eval", steps=500)
# experiment.launch_tensorboard()
# experiment.export_model()
