import comet_ml
import tsaplay.constants as CONSTANTS
from tsaplay.datasets import Dataset
from tsaplay.embeddings import Embedding
from tsaplay.features import FeatureProvider
from tsaplay.experiments import Experiment
from tsaplay.models import LCRRot


def main():

    embedding = Embedding(CONSTANTS.GLOVE_WIKI_GIGA_300)

    dong = Dataset(*CONSTANTS.DONG)
    restaurants = Dataset(*CONSTANTS.RESTAURANTS)
    laptops = Dataset(*CONSTANTS.LAPTOPS)
    wang = Dataset(*CONSTANTS.WANG)

    FeatureProvider([dong, restaurants, laptops, wang], embedding)

    # feature_provider = FeatureProvider([dong], embedding)

    # model = LCRRot(
    #     params={"hidden_units": 5, "batch-size": 25},
    #     aux_config={"n_attn_heatmaps": 2},
    # )

    # experiment = Experiment(
    #     feature_provider, model, config={"tf_random_seed": 1234}
    # )

    # experiment.run(job="train+eval", steps=5)


if __name__ == "__main__":
    main()
