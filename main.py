from tsaplay.datasets.Dataset import Dataset, DATASETS
from tsaplay.embeddings.Embedding import Embedding
from tsaplay.features.FeatureProvider import FeatureProvider
from tsaplay.experiments.Experiment import Experiment

# from tsaplay.models.Tang2016a.Lstm import Lstm
# from tsaplay.models.Tang2016a.TdLstm import TdLstm
# from tsaplay.models.Tang2016a.TcLstm import TcLstm
# from tsaplay.models.Zheng2018.LcrRot import LcrRot

# from tsaplay.models.Tang2016b.MemNet import MemNet
# from tsaplay.models.Chen2017.RecurrentAttentionNetwork import (
#     RecurrentAttentionNetwork
# )
# from tsaplay.models.Ma2017.InteractiveAttentionNetwork import (
#     InteractiveAttentionNetwork
# )

debug = Dataset(*DATASETS.DEBUG)
dong = Dataset(*DATASETS.DONG)
glv_wiki = Embedding("glove-wiki-gigaword-50")

feature_provider = FeatureProvider(datasets=[dong, debug], embedding=glv_wiki)

# experiment = Experiment(feature_provider, LcrRot())

# experiment.run(job="train+eval", steps=1)
# experiment.launch_tensorboard()
# experiment.export_model(restart_tfserve=True)
