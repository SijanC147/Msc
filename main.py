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
    # datasets = [Dataset(*DATASETS.DONG)]
    # embedding = Embedding(EMBEDDINGS.GLOVE_TWITTER_200)
    datasets = [Dataset(*DATASETS.DEBUG)]
    embedding = Embedding(EMBEDDINGS.GLOVE_WIKI_GIGA_50)

    feature_provider = FeatureProvider(datasets, embedding)

    model = LCRRot({"batch-size": 5, "hidden_units": 5})

    experiment = Experiment(
        feature_provider,
        model,
        config={
            "tf_random_seed": 1234,
            "save_checkpoints_steps": 5,
            "save_summary_steps": 1,
        },
        contd_tag="testing-context-switch-23",
    )

    experiment.run(job="train+eval", steps=15)
    # experiment.launch_tensorboard()
    # experiment.export_model()


if __name__ == "__main__":
    main()
