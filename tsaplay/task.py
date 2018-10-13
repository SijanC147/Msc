import argparse
import comet_ml
import tensorflow as tf
from tsaplay.datasets import Dataset
from tsaplay.embeddings import Embedding
from tsaplay.features import FeatureProvider
from tsaplay.experiments import Experiment
import tsaplay.models as tsa_models
import tsaplay.constants as CNSTS
from tsaplay.utils.io import cprnt

from tsaplay.datasets import (
    DEBUG,
    DEBUGV2,
    RESTAURANTS,
    LAPTOPS,
    DONG,
    NAKOV,
    ROSENTHAL,
    SAEIDI,
    WANG,
    XUE,
)

from tsaplay.embeddings import (
    FASTTEXT_WIKI_300,
    GLOVE_TWITTER_25,
    GLOVE_TWITTER_50,
    GLOVE_TWITTER_100,
    GLOVE_TWITTER_200,
    GLOVE_WIKI_GIGA_50,
    GLOVE_WIKI_GIGA_100,
    GLOVE_WIKI_GIGA_200,
    GLOVE_WIKI_GIGA_300,
    GLOVE_COMMON42_300,
    GLOVE_COMMON840_300,
    W2V_GOOGLE_300,
    W2V_RUS_300,
)


DATASETS = {
    "debug": DEBUG,
    "debugv2": DEBUGV2,
    "restaurants": RESTAURANTS,
    "laptops": LAPTOPS,
    "dong": DONG,
    "nakov": NAKOV,
    "rosenthal": ROSENTHAL,
    "saeidi": SAEIDI,
    "wang": WANG,
    "xue": XUE,
}

EMBEDDINGS = {
    "fasttext": FASTTEXT_WIKI_300,
    "twitter-25": GLOVE_TWITTER_25,
    "twitter-50": GLOVE_TWITTER_50,
    "twitter-100": GLOVE_TWITTER_100,
    "twitter-200": GLOVE_TWITTER_200,
    "wiki-50": GLOVE_WIKI_GIGA_50,
    "wiki-100": GLOVE_WIKI_GIGA_100,
    "wiki-200": GLOVE_WIKI_GIGA_200,
    "wiki-300": GLOVE_WIKI_GIGA_300,
    "commoncrawl-42": GLOVE_COMMON42_300,
    "commoncrawl-840": GLOVE_COMMON840_300,
    "w2v-google-300": W2V_GOOGLE_300,
    "w2v-rus-300": W2V_RUS_300,
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
    tf.logging.set_verbosity(args.verbosity)

    cprnt(bow="JOB DIR: {0}".format(args.job_dir))
    # embedding = Embedding(EMBEDDINGS.get(args.embedding))

    # datasets = [Dataset(*DATASETS.get(dataset)) for dataset in args.datasets]

    # feature_provider = FeatureProvider(datasets, embedding)

    # model = MODELS.get(args.model)(params={"batch-size": args.batch_size})

    # experiment = Experiment(
    #     feature_provider, model, contd_tag=args.contd_tag, job_dir=args.job_dir
    # )

    # experiment.run(job="train+eval", steps=args.steps)


if __name__ == "__main__":

    cprnt("PACKAGE PATH: {0}".format(CNSTS.PACKAGE_PATH))
    cprnt("HOME PATH: {0}".format(CNSTS.HOME_PATH))
    cprnt("DATA PATH: {0}".format(CNSTS.DATA_PATH))
    cprnt("ASSETS PATH: {0}".format(CNSTS.ASSETS_PATH))
    cprnt("DEFAULT FONT PATH: {0}".format(CNSTS.DEFAULT_FONT_PATH))
    cprnt("SPACY MODEL PATH: {0}".format(CNSTS.SPACY_MODEL_PATH))
    cprnt("DATASET PATH: {0}".format(CNSTS.DATASET_DATA_PATH))
    cprnt("EMBEDDING PATH: {0}".format(CNSTS.EMBEDDING_DATA_PATH))
    cprnt("FEATURES PATH: {0}".format(CNSTS.FEATURES_DATA_PATH))
    cprnt("EXPERIMENT PATH: {0}".format(CNSTS.EXPERIMENT_DATA_PATH))
    cprnt("EXPORTS PATH: {0}".format(CNSTS.EXPORTS_DATA_PATH))

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
        "--job-dir",
        help="GCS location to write checkpoints and export models",
        # required=True,
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
