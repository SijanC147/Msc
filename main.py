import comet_ml
from tsaplay.datasets.Dataset import Dataset, DATASETS
from tsaplay.embeddings.Embedding import Embedding, EMBEDDINGS
from tsaplay.features.FeatureProvider import FeatureProvider
from tsaplay.experiments.Experiment import Experiment
from tsaplay.models.LCRRot import LCRRot

from tsaplay.models.Lstm import Lstm

from tsaplay.models.TCLstm import TCLstm

# from tsaplay.models.TDLstm import TDLstm
# from tsaplay.models.Ian import Ian
# from tsaplay.models.MemNet import MemNet
# from tsaplay.models.Ram import Ram


def main():
    datasets = [Dataset(*DATASETS.DONG)]
    embedding = Embedding(EMBEDDINGS.GLOVE_TWITTER_100)

    feature_provider = FeatureProvider(datasets, embedding)

    model = TCLstm()

    experiment = Experiment(
        feature_provider,
        model,
        config={"tf_random_seed": 1234},
        contd_tag="dong_100_emb",
    )

    experiment.run(job="train+eval", steps=1000)
    experiment.launch_tensorboard()
    # experiment.export_model()


if __name__ == "__main__":
    main()
