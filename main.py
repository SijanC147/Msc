from tensorflow.estimator import RunConfig  # pylint: disable=E0401
from tsaplay.datasets.Dataset import Dataset, DATASETS
from tsaplay.embeddings.Embedding import Embedding
from tsaplay.features.FeatureProvider import FeatureProvider
from tsaplay.experiments.Experiment import Experiment

# from tsaplay.models.Lstm import Lstm
# from tsaplay.models.TCLstm import TCLstm
# from tsaplay.models.TDLstm import TDLstm
# from tsaplay.models.LCRRot import LCRRot
# from tsaplay.models.Ian import Ian
from tsaplay.models.MemNet import MemNet

from tsaplay.models.Ram import Ram

debug = Dataset(*DATASETS.DEBUG)
glv = Embedding("glove-wiki-gigaword-50")
model = MemNet(run_config=RunConfig(tf_random_seed=1234))

feature_provider = FeatureProvider(datasets=[debug], embedding=glv)

experiment = Experiment(feature_provider, model)
experiment.run(job="train+eval", steps=10)
experiment.launch_tensorboard()
# experiment.export_model()
