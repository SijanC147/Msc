import argparse
import traceback
from inspect import isclass, isabstract, getmembers
from importlib import import_module
from sys import argv
from os import environ, execvpe
from os.path import join
import pkg_resources as pkg
import comet_ml  # pylint: disable=W0611
import tensorflow as tf
from tsaplay.utils.debug import timeit
from tsaplay.utils.io import (
    args_to_dict,
    arg_with_list,
    datasets_cli_arg,
    list_folders,
    cprnt,
)
from tsaplay.utils.data import corpora_vocab
from tsaplay.datasets import Dataset
from tsaplay.embeddings import Embedding
from tsaplay.features import FeatureProvider
from tsaplay.experiments import Experiment
from tsaplay.models.tsa_model import TsaModel
from tsaplay.constants import EMBEDDING_SHORTHANDS, ASSETS_PATH, MODELS_PATH


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
        "--new",
        "-n",
        help="Reset batch counter environment variable.",
        action="store_true",
        default=False,
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
        choices=[
            f.split(".")[0]
            for f in list_folders(MODELS_PATH, kind="files")
            if f.endswith(".py") and not f.startswith("__init__")
        ],
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
        default=None,
    )

    single_task_parser.add_argument(
        "--steps",
        "-s",
        type=int,
        help="Choose how long to train the model in steps",
        default=None,
    )

    single_task_parser.add_argument(
        "--epochs",
        "-e",
        type=int,
        help="Choose how long to train the model in epochs",
        default=None,
    )

    single_task_parser.add_argument(
        "--verbosity",
        "-v",
        choices=["DEBUG", "INFO", "WARN", "ERROR"],
        default="ERROR",
        help="Set logging verbosity",
    )

    return parser


def load_model(model_name, container=None):
    module_name = "{container}.{model_name}".format_map(
        {
            "container": (container or "tsaplay.models"),
            "model_name": model_name,
        }
    )
    try:
        module = import_module(module_name)
    except ModuleNotFoundError:
        raise ValueError("Could not resolve {}".format(module_name))

    model = getmembers(
        module,
        predicate=(
            lambda attr: isclass(attr)
            and not isabstract(attr)
            and issubclass(attr, TsaModel)
        ),
    )
    if len(model) == 1:
        return model[0][1]
    raise ValueError("Found no valid models in {}".format(module_name))


def make_feature_provider(args):
    datasets = [Dataset(name, redist) for (name, redist) in args.datasets]

    embedding_name, filters_arg = args.embedding

    filters = [f for f in filters_arg if f != "corpus"] if filters_arg else []
    if filters_arg and "corpus" in filters_arg:
        corpora = ({**ds.train_corpus, **ds.test_corpus} for ds in datasets)
        filters += [corpora_vocab(*corpora)]

    embedding = Embedding(EMBEDDING_SHORTHANDS.get(embedding_name), filters)
    params = args_to_dict(args.model_params)

    return FeatureProvider(datasets, embedding, **params)


@timeit("Starting Experiment", "Experiment complete")
def run_experiment(args, experiment_index=None):
    if experiment_index is not None:
        cprnt(y="Running experiment {}".format(experiment_index + 1))
        cprnt(y="Args: {}".format(args))

    tf.logging.set_verbosity(args.verbosity)

    feature_provider = make_feature_provider(args)

    params = args_to_dict(args.model_params)
    aux_config = args_to_dict(args.aux_config)

    if args.batch_size:
        params.update({"batch-size": args.batch_size})
    model = load_model(args.model)(params, aux_config)

    model.params["steps"] = (
        args.steps if args.steps is not None else model.params.get("steps")
    )
    model.params["epochs"] = (
        args.epochs if args.epochs is not None else model.params.get("epochs")
    )
    epoch_steps, num_training_samples = feature_provider.steps_per_epoch(
        model.params["batch-size"]
    )
    model.params.update(
        {"epoch_steps": epoch_steps, "shuffle_buffer": num_training_samples}
    )
    if args.steps is not None:
        model.params.pop("epochs", None)
    if model.params.get("epochs") is not None:
        model.params.pop("steps", None)

    experiment = Experiment(
        feature_provider,
        model,
        run_config=args_to_dict(args.run_config),
        comet_api=args.comet_api,
        comet_workspace=args.comet_workspace,
        contd_tag=args.contd_tag,
        job_dir=args.job_dir,
    )

    experiment.run(job="train+eval", steps=args.steps, epochs=args.epochs)


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
    task_parser = argument_parser()
    tasks = parse_batch_file(batch_file_path, defaults)
    if job_dir:
        tasks = [t + ["--job-dir", job_dir] for t in tasks]
    task_index = int(environ.get("TSATASK", 0))
    cprnt(warn="TASK INDEX: {0}".format(task_index))
    if task_index >= len(tasks):
        del environ["TSATASK"]
        return
    try:
        task_args = task_parser.parse_args(tasks[task_index])
        cprnt(info="RUNNING TASK {0}: {1}".format(task_index, task_args))
        run_experiment(task_args, experiment_index=task_index)
    except Exception:  # pylint: disable=W0703
        traceback.print_exc()
    environ["TSATASK"] = str(task_index + 1)
    job_dir_arg = "--job-dir {}".format(job_dir) if (job_dir) else ""
    defaults_arg = (
        "--defaults {}".format(" ".join(defaults)) if defaults else ""
    )
    next_cmd = """python3 -m tsaplay.task batch {batch_file} {job_dir} {defaults}""".format(
        batch_file=batch_file_path, job_dir=job_dir_arg, defaults=defaults_arg
    )
    execvpe("python3", next_cmd.split(), environ)


def main():
    nvidia_cuda_prof_tools_path_fix()
    args = argument_parser().parse_args()
    try:
        new_batch = args.new
    except AttributeError: # ? Not running in batch mode.
        new_batch = False
    if new_batch and environ.get("TSATASK") is not None:
        del environ["TSATASK"]
    try:
        if args.defaults and "--job-dir" in args.defaults:
            def_job_dir_index = args.defaults.index("--job-dir")
            job_dir = args.job_dir or args.defaults[def_job_dir_index + 1]
            defaults = (
                args.defaults[:(def_job_dir_index)]
                + args.defaults[(def_job_dir_index + 2) :]
            )
            run_next_experiment(args.batch_file, job_dir, defaults)
        else:
            run_next_experiment(args.batch_file, args.job_dir, args.defaults)
    except AttributeError: # ? Not running in batch mode.
        run_experiment(args)
    pkg.cleanup_resources()


if __name__ == "__main__":
    main()
