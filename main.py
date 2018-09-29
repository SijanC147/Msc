from tsaplay.datasets.Dataset import Dataset, DATASETS
from tsaplay.datasets.CompoundDataset import CompoundDataset
from tsaplay.embeddings.Embedding import Embedding
from tsaplay.embeddings.PartialEmbedding import PartialEmbedding
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

# dataset = Dataset(path=DATASETS.DEBUG_PATH, parser=DATASETS.DEBUG_PARSER)
# embedding = PartialEmbedding(dataset.name, dataset.corpus, "glove-twitter-25") # noqa
dong = Dataset(path=DATASETS.DONG2014_PATH, parser=DATASETS.DONG2014_PARSER)
xue = Dataset(path=DATASETS.XUE2018_PATH, parser=DATASETS.XUE2018_PARSER)
compound = CompoundDataset(dong, xue)

glv_twitter = Embedding("glove-twitter-200")

feature_provider = FeatureProvider(dataset=compound, embedding=glv_twitter)
experiment = Experiment(feature_provider, LcrRot())

experiment.run(job="train+eval", steps=1)
# experiment.launch_tensorboard()
# experiment.export_model(restart_tfserve=True)
