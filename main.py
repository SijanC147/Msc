import comet_ml
from tsaplay.datasets.Dataset import Dataset, DATASETS
from tsaplay.embeddings.Embedding import Embedding, EMBEDDINGS
from tsaplay.features.FeatureProvider import FeatureProvider
from tsaplay.experiments.Experiment import Experiment

# from tsaplay.models.Lstm import Lstm
# from tsaplay.models.TCLstm import TCLstm
# from tsaplay.models.TDLstm import TDLstm
from tsaplay.models.LCRRot import LCRRot

# from tsaplay.models.Ian import Ian
from tsaplay.models.MemNet import MemNet

from tsaplay.models.Ram import Ram

dataset = Dataset(*DATASETS.DEBUG)
glv = Embedding(EMBEDDINGS.GLOVE_WIKI_GIGA_50)
# model = LCRRot(
#     params={"n_attn_heatmaps": 2, "hidden_units": 5, "batch_size": 5}
# )
model = MemNet(
    params={"n_attn_heatmaps": 2, "hidden_units": 5, "batch_size": 5}
)
feature_provider = FeatureProvider(datasets=[dataset], embedding=glv)

experiment = Experiment(
    feature_provider, model, config={"tf_random_seed": 1234}
)
experiment.run(job="train+eval", steps=1)
# experiment.launch_tensorboard()
# experiment.export_model()
