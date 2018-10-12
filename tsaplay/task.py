import argparse
import comet_ml
import tensorflow as tf
import tsaplay.constants as CONSTANTS
from tsaplay.datasets import Dataset
from tsaplay.embeddings import Embedding
from tsaplay.features import FeatureProvider
from tsaplay.experiments import Experiment
import tsaplay.models as tsa_models

DATASETS = {
    "debug": CONSTANTS.DEBUG,
    "debugv2": CONSTANTS.DEBUGV2,
    "restaurants": CONSTANTS.RESTAURANTS,
    "laptops": CONSTANTS.LAPTOPS,
    "dong": CONSTANTS.DONG,
    "nakov": CONSTANTS.NAKOV,
    "rosenthal": CONSTANTS.ROSENTHAL,
    "saeidi": CONSTANTS.SAEIDI,
    "wang": CONSTANTS.WANG,
    "xue": CONSTANTS.XUE,
}

EMBEDDINGS = {
    "fasttext": CONSTANTS.FASTTEXT_WIKI_300,
    "twitter-25": CONSTANTS.GLOVE_TWITTER_25,
    "twitter-50": CONSTANTS.GLOVE_TWITTER_50,
    "twitter-100": CONSTANTS.GLOVE_TWITTER_100,
    "twitter-200": CONSTANTS.GLOVE_TWITTER_200,
    "wiki-50": CONSTANTS.GLOVE_WIKI_GIGA_50,
    "wiki-100": CONSTANTS.GLOVE_WIKI_GIGA_100,
    "wiki-200": CONSTANTS.GLOVE_WIKI_GIGA_200,
    "wiki-300": CONSTANTS.GLOVE_WIKI_GIGA_300,
    "commoncrawl-42": CONSTANTS.GLOVE_COMMON42_300,
    "commoncrawl-840": CONSTANTS.GLOVE_COMMON840_300,
    "w2v-google-300": CONSTANTS.W2V_GOOGLE_300,
    "w2v-rus-300": CONSTANTS.W2V_RUS_300,
}

MODELS = {
    "lstm": tsa_models.Lstm,
    "tdlstm": tsa_models.TDLstm,
    "tclstm": tsa_models.TCLstm,
    "lcrrot": tsa_models.LCRRot,
    "ian": tsa_models.Ian,
    "memnet": tsa_models.MemNet,
    "ram": tsa_models.Ram,
}


def run_experiment(args):
    print(args)
    tf.logging.set_verbosity(args.verbosity)

    embedding = Embedding(EMBEDDINGS.get(args.embedding))

    datasets = [Dataset(*DATASETS.get(dataset)) for dataset in args.datasets]

    feature_provider = FeatureProvider(datasets, embedding)

    model = MODELS.get(args.model)(params={"batch-size": args.batch_size})

    experiment = Experiment(feature_provider, model, contd_tag=args.contd_tag)

    experiment.run(job="train+eval", steps=args.steps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--embedding",
        "-em",
        type=str,
        choices=[*EMBEDDINGS],
        help="Pre-trained embedding to use.",
        default="wiki-50",
    )

    parser.add_argument(
        "--datasets",
        "-ds",
        type=str,
        choices=[*DATASETS],
        help="One or more datasets to use for training and evaluation.",
        default=["dong"],
        nargs="+",
    )

    parser.add_argument(
        "--model",
        "-m",
        type=str,
        choices=[*MODELS],
        help="Choose model to train",
        default="lcrrot",
    )

    parser.add_argument(
        "--contd-tag",
        type=str,
        help="Continue a specific experiment resolved through this tag",
    )

    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        help="Size of training and evaluation batches",
        default=25,
    )

    parser.add_argument(
        "--steps",
        "-s",
        type=int,
        help="Choose how long to train the model",
        default=300,
    )

    parser.add_argument(
        "--verbosity",
        "-v",
        choices=["DEBUG", "ERROR", "FATAL", "INFO", "WARN"],
        default="INFO",
        help="Set logging verbosity",
    )

    run_experiment(parser.parse_args())
