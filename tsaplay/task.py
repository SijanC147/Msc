import argparse
from os import environ, getcwd
import comet_ml
import tensorflow as tf
import pkg_resources as pkg
from tsaplay.utils.io import cprnt
from tsaplay.utils.decorators import timeit
from tsaplay.datasets import Dataset
from tsaplay.embeddings import Embedding
from tsaplay.features import FeatureProvider
from tsaplay.experiments import Experiment
import tsaplay.models as tsa_models
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
    "tdlstm": tsa_models.TdLstm,
    "tclstm": tsa_models.TcLstm,
    "lcrrot": tsa_models.LcrRot,
    "ian": tsa_models.Ian,
    "memnet": tsa_models.MemNet,
    "ram": tsa_models.Ram,
}


def argument_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--job-dir",
        help="GCS location to write checkpoints to and export models",
    )

    parser.add_argument(
        "--embedding",
        "-em",
        type=str,
        choices=[*EMBEDDINGS],
        help="Pre-trained embedding to use.",
        default="wiki-50",
    )

    parser.add_argument(
        "--filter-embedding",
        "-fe",
        help="Filter embedding on unique tokens from datasets",
        action="store_true",
    )

    parser.add_argument(
        "--comet-api",
        "-cmt",
        help="Comet.ml API key to upload experiment, contd_tag must be set",
    )

    parser.add_argument(
        "--datasets",
        "-ds",
        type=str,
        choices=Dataset.list_installed_datasets(),
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
        "--model-params",
        "-mp",
        nargs="*",
        help="H-Params to forward to model (space-delimted <key>=<value>)",
        required=False,
    )

    parser.add_argument(
        "--aux-config",
        "-aux",
        nargs="*",
        help="AUX config to forward to model (space-delimted <key>=<value>)",
        required=False,
    )

    parser.add_argument(
        "--run-config",
        "-rc",
        nargs="*",
        help="Custom run_config parameters (space-delimted <key>=<value>)",
        required=False,
    )

    parser.add_argument(
        "--contd-tag",
        "-contd",
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

    return parser


def args_to_dict(args):
    args_dict = args or {}
    if args_dict:
        args_dict = [arg.split("=") for arg in args_dict]
        args_dict = {
            arg[0]: (
                int(arg[1])
                if arg[1].isdigit()
                else float(arg[1])
                if arg[1].replace(".", "", 1).isdigit()
                else True
                if arg[1].lower() == "true"
                else False
                if arg[1].lower() == "false"
                else arg[1]
            )
            for arg in args_dict
        }
    return args_dict


def get_feature_provider(args):
    datasets = [Dataset(name) for name in args.datasets]

    emb_filter = (
        [list(set(sum([ds.corpus for ds in datasets], [])))]
        if args.filter_embedding
        else None
    )
    embedding = Embedding(EMBEDDINGS.get(args.embedding), filters=emb_filter)

    return FeatureProvider(datasets, embedding)


def run_experiment(args):
    tf.logging.set_verbosity(args.verbosity)

    feature_provider = get_feature_provider(args)

    params = args_to_dict(args.model_params)
    params.update({"batch-size": args.batch_size})
    model = MODELS.get(args.model)(params, args_to_dict(args.aux_config))

    experiment = Experiment(
        feature_provider,
        model,
        run_config=args_to_dict(args.run_config),
        comet_api=args.comet_api,
        contd_tag=args.contd_tag,
        job_dir=args.job_dir,
    )

    experiment.run(job="train+eval", steps=args.steps)

    pkg.cleanup_resources()


@timeit("Starting task", "Task complete")
def main():
    parser = argument_parser()
    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
