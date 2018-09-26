from tsaplay.datasets.Dataset import Dataset, DATASETS
from tsaplay.embeddings.Embedding import Embedding, EMBEDDINGS
from tsaplay.embeddings.PartialEmbedding import PartialEmbedding
from tsaplay.features.FeatureProvider import FeatureProvider
from tsaplay.experiments.Experiment import Experiment

dataset = Dataset(path=DATASETS.DEBUG_PATH, parser=DATASETS.DEBUG_PARSER)
