import comet_ml
from tsaplay.constants import GLOVE_WIKI_GIGA_50, DONG
from tsaplay.datasets.Dataset import Dataset
from tsaplay.embeddings.Embedding import Embedding
from tsaplay.features.FeatureProvider import FeatureProvider
from tsaplay.experiments.Experiment import Experiment
from tsaplay.models import LCRRot


def main():

    embedding = Embedding(GLOVE_WIKI_GIGA_50)

    dong = Dataset(*DONG, distribution=[0.33, 0.33, 0.34])

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
