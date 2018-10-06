import comet_ml
from tsaplay.datasets.Dataset import Dataset, DATASETS
from tsaplay.embeddings.Embedding import Embedding, EMBEDDINGS
from tsaplay.features.FeatureProvider import FeatureProvider
from tsaplay.experiments.Experiment import Experiment

# from tsaplay.models.Lstm import Lstm
# from tsaplay.models.TCLstm import TCLstm
# from tsaplay.models.TDLstm import TDLstm
from tsaplay.models.LCRRot import LCRRot

# from tsaplay.models.Ian import Ian
from tsaplay.models.MemNet import MemNet

from tsaplay.models.Ram import Ram


def main():
    datasets = [Dataset(*DATASETS.DONG)]
    embedding = Embedding(EMBEDDINGS.GLOVE_TWITTER_200)

    feature_provider = FeatureProvider(datasets, embedding)

    model = LCRRot()

    experiment = Experiment(
        feature_provider, model, contd_tag="dong-twitter-200-200hu-div"
    )

    experiment.run(job="train+eval", steps=600)
    # experiment.launch_tensorboard()
    # experiment.export_model()


if __name__ == "__main__":
    main()
