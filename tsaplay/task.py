import argparse
import traceback
from sys import argv
from os import environ, execvpe
from os.path import join
import pkg_resources as pkg
from datetime import datetime
import comet_ml  # pylint: disable=W0611
import tensorflow as tf
from tsaplay.utils.debug import timeit, cprnt
from tsaplay.utils.io import args_to_dict, arg_with_list, datasets_cli_arg
from tsaplay.utils.data import corpora_vocab
from tsaplay.datasets import Dataset
from tsaplay.embeddings import Embedding
from tsaplay.features import FeatureProvider
from tsaplay.experiments import Experiment
import tsaplay.models as tsa_models
from tsaplay.constants import EMBEDDING_SHORTHANDS, ASSETS_PATH, RANDOM_SEED

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

    subparsers = parser.add_subparsers()
    batch_task_parser = subparsers.add_parser("batch")
    batch_task_parser.add_argument(
        "batch_file",
        help="Path to batch file containing list of jobs to run sequentially",
    )
    batch_task_parser.add_argument(
        "--job-dir",
        help="GCS location to write checkpoints to and export models",
    )
    batch_task_parser.add_argument(
        "--defaults",
        help="Default parameters to forward to all tasks (low precedence)",
        nargs=argparse.REMAINDER,
    )

    single_task_parser = subparsers.add_parser("single")
    single_task_parser.add_argument(
        "--job-dir",
        help="GCS location to write checkpoints to and export models",
    )

    single_task_parser.add_argument(
        "--embedding",
        "-em",
        type=arg_with_list,
        help="Pre-trained embedding to use, <embedding>[..filters]",
        default="wiki-50",
    )

    single_task_parser.add_argument(
        "--datasets",
        "-ds",
        type=datasets_cli_arg,
        help="One or more datasets to use, <dataset>[redist(,test_dist)]+",
        default=["dong"],
        nargs="+",
    )

    single_task_parser.add_argument(
        "--max-shards",
        "-ms",
        help="Max number of shards to partition embedding.",
        type=int,
    )

    single_task_parser.add_argument(
        "--comet-api",
        "-cmt",
        help="Comet.ml API key to upload experiment, contd_tag must be set",
    )

    single_task_parser.add_argument(
        "--comet-workspace",
        "-wrk",
        help="Comet.ml workspace to use, comet-api must be set",
    )

    single_task_parser.add_argument(
        "--model",
        "-m",
        type=str,
        choices=[*MODELS],
        help="Choose model to train",
        default="lcrrot",
    )

    single_task_parser.add_argument(
        "--model-params",
        "-mp",
        nargs="*",
        action="append",
        help="H-Params to forward to model (space-delimited <key>=<value>)",
        required=False,
    )

    single_task_parser.add_argument(
        "--aux-config",
        "-aux",
        action="append",
        nargs="*",
        help="AUX config to forward to model (space-delimited <key>=<value>)",
        required=False,
    )

    single_task_parser.add_argument(
        "--run-config",
        "-rc",
        nargs="*",
        action="append",
        help="Custom run_config parameters (space-delimited <key>=<value>)",
        required=False,
    )

    single_task_parser.add_argument(
        "--contd-tag",
        "-contd",
        type=str,
        help="Continue a specific experiment resolved through this tag",
    )

    single_task_parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        help="Size of training and evaluation batches",
        default=25,
    )

    single_task_parser.add_argument(
        "--steps",
        "-s",
        type=int,
        help="Choose how long to train the model",
        default=300,
    )

    single_task_parser.add_argument(
        "--verbosity",
        "-v",
        choices=["DEBUG", "INFO", "WARN", "ERROR", "FATAL"],
        default="INFO",
        help="Set logging verbosity",
    )

    return parser


def make_feature_provider(args):
    datasets = [Dataset(name, redist) for (name, redist) in args.datasets]

    embedding_name, filters_arg = args.embedding

    filters = [f for f in filters_arg if f != "corpus"] if filters_arg else []
    if filters_arg and "corpus" in filters_arg:
        corpora = ({**ds.train_corpus, **ds.test_corpus} for ds in datasets)
        filters += [corpora_vocab(*corpora)]

    embedding = Embedding(EMBEDDING_SHORTHANDS.get(embedding_name), filters)
    params = args_to_dict(
        args.model_params
    )  # TODO: add parameters for the random_uniform init bounds for oov tokens

    return FeatureProvider(datasets, embedding, **params)


@timeit("Starting Experiment", "Experiment complete")
def run_experiment(args, experiment_index=None):
    if experiment_index is not None:
        cprnt(y="Running experiment {}".format(experiment_index+1))
        cprnt(y="Args: {}".format(args))

    tf.logging.set_verbosity(args.verbosity)

    feature_provider = make_feature_provider(args)

    params = args_to_dict(args.model_params)
    params.update({"batch-size": args.batch_size})
    model = MODELS.get(args.model)(params, args_to_dict(args.aux_config))

    run_config = args_to_dict(args.run_config)
    experiment = Experiment(
        feature_provider,
        model,
        run_config=run_config,
        comet_api=args.comet_api,
        comet_workspace=args.comet_workspace,
        contd_tag=args.contd_tag,
        job_dir=args.job_dir,
    )

    experiment.run(job="train+eval", steps=args.steps)


def nvidia_cuda_prof_tools_path_fix():
    ld_lib_path = environ.get("LD_LIBRARY_PATH")
    if ld_lib_path and ld_lib_path.find("CUPTI") == -1:
        cupti_path = "/usr/local/cuda-9.0/extras/CUPTI/lib64"
        environ["LD_LIBRARY_PATH"] = ":".join([cupti_path, ld_lib_path])
        execvpe("python3", ["python3"] + argv, environ)


def parse_batch_file(batch_file_path, defaults=None):
    try:
        batch_file = open(batch_file_path, "r")
    except FileNotFoundError:
        batch_file = open(join(ASSETS_PATH, batch_file_path), "r")
    with batch_file:
        return [
            ["single"] + (defaults or []) + cmd.strip().split()
            for cmd in batch_file
            if cmd.strip() and cmd[0] not in ["#", ";"]
        ]


def run_next_experiment(batch_file_path, job_dir=None, defaults=None):
    tasks = parse_batch_file(batch_file_path, defaults)
    if job_dir:
        tasks = [t + ["--job-dir", job_dir] for t in tasks]
    task_index = int(environ.get("TSATASK", 0))
    if task_index >= len(tasks):
        del environ["TSATASK"]
        return
    task_parser = argument_parser()
    try:
        task_args = task_parser.parse_args(tasks[task_index])
        cprnt("RUNNING TASK {0}: {1}".format(task_index, task_args))
        run_experiment(task_args, experiment_index=task_index)
    except Exception:  # pylint: disable=W0703
        traceback.print_exc()
    environ["TSATASK"] = str(task_index + 1)
    job_dir_arg = "--job-dir {}".format(job_dir) if job_dir else ""
    defaults_arg = (
        "--defaults {}".format(" ".join(defaults)) if defaults else ""
    )
    next_cmd = "python3 -m tsaplay.task batch {batch_file} {job_dir} {defaults}".format(
        batch_file=batch_file_path, job_dir=job_dir_arg, defaults=defaults_arg
    )
    execvpe("python3", next_cmd.split(), environ)


def main():
    nvidia_cuda_prof_tools_path_fix()
    args = argument_parser().parse_args()
    try:
        try:
            run_next_experiment(args.batch_file, args.job_dir, args.defaults)
        except AttributeError:
            run_next_experiment(args.batch_file, defaults=args.defaults)
    except AttributeError:
        run_experiment(args)
    pkg.cleanup_resources()


if __name__ == "__main__":
    main()
