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

    embedding = Embedding(EMBEDDINGS.GLOVE_WIKI_GIGA_50)

    dong = Dataset(*DATASETS.DONG, distribution=[0.33, 0.33, 0.34])

    feature_provider = FeatureProvider([dong], embedding)

    model = LCRRot(
        params={"hidden_units": 5, "batch-size": 25},
        aux_config={"n_attn_heatmaps": 2},
    )

    experiment = Experiment(
        feature_provider, model, config={"tf_random_seed": 1234}
    )

    experiment.run(job="train+eval", steps=5)


if __name__ == "__main__":
    main()
