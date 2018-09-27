from tsaplay.datasets.Dataset import Dataset, DATASETS
from tsaplay.embeddings.PartialEmbedding import PartialEmbedding
from tsaplay.features.FeatureProvider import FeatureProvider
from tsaplay.experiments.Experiment import Experiment
from tsaplay.models.Zheng2018.LcrRot import LcrRot

dataset = Dataset(path=DATASETS.DEBUG_PATH, parser=DATASETS.DEBUG_PARSER)
embedding = PartialEmbedding(dataset.name, dataset.corpus, "glove-twitter-25")
feature_provider = FeatureProvider(dataset, embedding)
experiment = Experiment(feature_provider, LcrRot())

experiment.run(job="train+eval", steps=10)
