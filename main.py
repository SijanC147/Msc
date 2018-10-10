import comet_ml
from tsaplay.datasets.Dataset import Dataset, DATASETS
from tsaplay.embeddings.Embedding import Embedding, EMBEDDINGS
from tsaplay.features.FeatureProvider import FeatureProvider
from tsaplay.experiments.Experiment import Experiment
from tsaplay.models.LCRRot import LCRRot

# from tsaplay.models.Lstm import Lstm

# from tsaplay.models.TCLstm import TCLstm

# from tsaplay.models.TDLstm import TDLstm
# from tsaplay.models.Ian import Ian
# from tsaplay.models.MemNet import MemNet
# from tsaplay.models.Ram import Ram


def main():
    # datasets = [Dataset(*DATASETS.XUE)]
    # embedding = Embedding(EMBEDDINGS.GLOVE_WIKI_GIGA_300)
    # datasets = [Dataset(*DATASETS.DEBUGV2)]
    # embedding = Embedding(EMBEDDINGS.GLOVE_WIKI_GIGA_50)

    # feature_provider = FeatureProvider(datasets, embedding)

    # model = LCRRot(params={"shuffle-buffer": 100000, "attn_heatmaps": False})

    # experiment = Experiment(
    #     feature_provider,
    #     model,
    #     config={"tf_random_seed": 1234},
    #     # contd_tag="larger shuffle buffer",
    # )

    # experiment.run(job="train+eval", steps=100)
    # experiment.launch_tensorboard()
    # experiment.export_model()
    distribution = {"train": [0.2, 0.7, 0.1], "test": [0.3, 0.4, 0.3]}
    redist_debugV2 = Dataset(*DATASETS.DEBUGV2, distribution)


if __name__ == "__main__":
    main()
