from tsaplay.datasets.Dataset import Dataset, DATASETS
from tsaplay.embeddings.Embedding import Embedding
from tsaplay.features.FeatureProvider import FeatureProvider
from tsaplay.experiments.Experiment import Experiment

# from tsaplay.models.Tang2016a.Lstm import Lstm
# from tsaplay.models.Tang2016a.TdLstm import TdLstm
# from tsaplay.models.Tang2016a.TcLstm import TcLstm
from tsaplay.models.Zheng2018.LcrRot import LcrRot

# from tsaplay.models.Tang2016b.MemNet import MemNet
# from tsaplay.models.Chen2017.RecurrentAttentionNetwork import (
#     RecurrentAttentionNetwork
# )
# from tsaplay.models.Ma2017.InteractiveAttentionNetwork import (
#     InteractiveAttentionNetwork
# )

xue = Dataset(*DATASETS.XUE)
dong = Dataset(*DATASETS.DONG)
glv = Embedding("glove-wiki-gigaword-300")

feature_provider = FeatureProvider(datasets=[dong], embedding=glv)

experiment = Experiment(feature_provider, LcrRot(), contd_tag="gold")
experiment.run(job="train+eval", steps=1000)
experiment.launch_tensorboard()
# experiment.export_model()
