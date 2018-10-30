import argparse
from sys import argv
from os import environ, getcwd, execvpe
from warnings import warn
import comet_ml
import tensorflow as tf
import pkg_resources as pkg
from tsaplay.utils.io import cprnt
from tsaplay.utils.decorators import timeit
import tsaplay.utils.filters as available_filters
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
        "--embedding-filters",
        "-ef",
        help="Filter embedding on unique tokens from datasets",
        nargs="*",
    )

    parser.add_argument(
        "--max-shards",
        "-ms",
        help="Max number of shards to partition embedding.",
        type=int,
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

    emb_filters = args.embedding_filters or []
    if emb_filters:
        emb_filters_resolved = []
        for emb_filter in emb_filters:
            if emb_filter == "corpus":
                emb_filters_resolved.append(
                    list(set(sum([ds.corpus for ds in datasets], [])))
                )
            else:
                try:
                    emb_filters_resolved.append(
                        available_filters.__getattribute__(emb_filter)
                    )
                except AttributeError:
                    warn("Could not find {} filter.".format(emb_filter))
    else:
        emb_filters_resolved = None

    max_shards = args.max_shards or 10
    embedding = Embedding(
        EMBEDDINGS.get(args.embedding),
        filters=emb_filters_resolved,
        max_shards=max_shards,
    )

    return FeatureProvider(datasets, embedding)


def run_experiment(args):
    tf.logging.set_verbosity(args.verbosity)

    feature_provider = get_feature_provider(args)

    params = args_to_dict(args.model_params)
    params.update({"batch-size": args.batch_size})
    model = MODELS.get(args.model)(params, args_to_dict(args.aux_config))

    run_config = args_to_dict(args.run_config)
    # distribute_config = tf.contrib.distribute.DistributeConfig(
    #     train_distribute=tf.contrib.distribute.ParameterServerStrategy(
    #         num_gpus_per_worker=4
    #     ),
    #     # eval_distribute=tf.contrib.distribute.ParameterServerStrategy(
    #     #     num_gpus_per_worker=4
    #     # ),
    #     # train_distribute=tf.contrib.distribute.MirroredStrategy(num_gpus=4),
    #     # eval_distribute=tf.contrib.distribute.MirroredStrategy(num_gpus=4),
    # )
    # run_config.update({"experimental_distribute": distribute_config})

    experiment = Experiment(
        feature_provider,
        model,
        run_config=run_config,
        comet_api=args.comet_api,
        contd_tag=args.contd_tag,
        job_dir=args.job_dir,
    )

    experiment.run(job="train+eval", steps=args.steps)

    pkg.cleanup_resources()


def nvidia_cuda_prof_tools_path_fix():
    ld_lib_path = environ.get("LD_LIBRARY_PATH")
    if ld_lib_path and ld_lib_path.find("CUPTI") == -1:
        cupti_path = "/usr/local/cuda-9.0/extras/CUPTI/lib64"
        environ["LD_LIBRARY_PATH"] = ":".join([cupti_path, ld_lib_path])
        execvpe("python3", ["python3"] + argv, environ)


@timeit("Starting task", "Task complete")
def main():
    parser = argument_parser()
    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    nvidia_cuda_prof_tools_path_fix()
    main()
