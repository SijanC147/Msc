import argparse
from os import system
from os.path import join, abspath
from json import dump
from setuptools import sandbox
from tsaplay.task import (
    make_feature_provider,
    argument_parser as task_argument_parser,
    parse_batch_file,
)
from tsaplay.utils.io import search_dir, copy, clean_dirs, args_to_dict
from tsaplay.constants import (
    ASSETS_PATH,
    DATASET_DATA_PATH,
    EMBEDDING_DATA_PATH,
    FEATURES_DATA_PATH,
)

# import tsaplay.models as tsa_models

# MODELS = {
#     "lstm": tsa_models.Lstm,
#     "tdlstm": tsa_models.TdLstm,
#     "tclstm": tsa_models.TcLstm,
#     "lcrrot": tsa_models.LcrRot,
#     "ian": tsa_models.Ian,
#     "memnet": tsa_models.MemNet,
#     "ram": tsa_models.Ram,
# }

DEST_PATH = join(ASSETS_PATH, "_{}")
DATASETS_DEST = DEST_PATH.format("datasets")
EMBEDDINGS_DEST = DEST_PATH.format("embeddings")
FEATURES_DEST = DEST_PATH.format("features")


def argument_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--jobs-file",
        "-jf",
        help="Text file with newline delimited args for multiple jobs",
    )

    parser.add_argument(
        "--job-id",
        "-jid",
        type=str,
        help="ID of the job to be submitted to gcloud.",
    )

    parser.add_argument(
        "--staging-bucket",
        "-bkt",
        type=str,
        default="tsaplay-bucket",
        help="Name of the gCloud staging bucket used to store experiments.",
    )

    parser.add_argument(
        "--job-dir",
        "-jdir",
        help="GCS location to write checkpoints to and export models",
    )

    parser.add_argument(
        "--job-labels",
        "-jlab",
        help="Labels to add to the job, must be in <key>=<value> format.",
        nargs="*",
        required=False,
    )

    parser.add_argument(
        "--machine-types",
        "-mch",
        help="Specify the gCloud machine types to use",
        nargs="*",
        required=False,
    )

    parser.add_argument(
        "--stream-logs",
        "-stream",
        help="Include flag to stream logs in gCloud submit command",
        action="store_true",
        required=False,
    )

    parser.add_argument(
        "--show-sdist",
        help="Print output from setup.py sdist",
        action="store_true",
    )

    parser.add_argument(
        "--task-args",
        "-t",
        help="Arguments to pass forward to the task module",
        nargs=argparse.REMAINDER,
        required=True,
    )

    return parser


def fix_requirements_for_machine_types(machine_types):
    with open("requirements.txt", "r") as requirements_file:
        reqs = requirements_file.read()
    reqs = reqs.replace("tensorflow-gpu==", "tensorflow==")
    for value in machine_types.values():
        if not isinstance(value, str):
            continue
        if "gpu" in value or "p100" in value or "v100" in value:
            reqs = reqs.replace("tensorflow==", "tensorflow-gpu==")
    with open("requirements.txt", "w") as requirements_file:
        requirements_file.write(reqs)


def write_gcloud_config(args):
    machine_types = args_to_dict(args.machine_types) or {
        "masterType": "standard",
        # "workerType": "n1-standard-16",
        # "parameterServerType": "standard_gpu",
        # "workerCount": 5,
        # "parameterServerCount": 2,
    }
    fix_requirements_for_machine_types(machine_types)
    gcloud_config = {
        "jobId": args.job_id,
        "labels": args_to_dict(args.job_labels),
        "trainingInput": {
            "scaleTier": "CUSTOM",
            **machine_types,
            "pythonVersion": "3.5",
            "runtimeVersion": "1.12",
            "region": "europe-west1",
            "args": args.task_args,
        },
    }
    config_file_path = join("gcp", "_config.json")
    with open(config_file_path, "w") as config_file:
        dump(gcloud_config, config_file, indent=4)


def parse_task_args(task_args):
    task_parser = task_argument_parser()
    parsed_args = task_parser.parse_args(task_args)
    try:
        jobs = parse_batch_file(
            parsed_args.batch_file, defaults=parsed_args.defaults
        )
        copy(parsed_args.batch_file, ASSETS_PATH)
    except AttributeError:
        jobs = [task_args]
    return [make_feature_provider(task_parser.parse_args(job)) for job in jobs]


def copy_dataset_files(datasets):
    for mode in ["train", "test"]:
        for dataset in datasets:
            srcdir_attr = "_{mode}_srcdir".format(mode=mode)
            dataset_srcdir = getattr(dataset, srcdir_attr)
            copy(
                dataset_srcdir,
                DATASETS_DEST,
                rel=DATASET_DATA_PATH,
                force=False,
                ignore="_redists",
            )


def copy_embedding_files(embedding):
    copy(embedding.gen_dir, EMBEDDINGS_DEST, rel=EMBEDDING_DATA_PATH)


def copy_feature_files(feature_provider):
    copy(
        feature_provider.gen_dir,
        FEATURES_DEST,
        rel=FEATURES_DATA_PATH,
        ignore="*.zip",
    )


def prepare_job_assets(args):
    feature_providers = parse_task_args(args.task_args)
    for feature_provider in feature_providers:
        copy_dataset_files(feature_provider.datasets)
        copy_embedding_files(feature_provider.embedding)
        copy_feature_files(feature_provider)

    sdist_args = ([] if args.show_sdist else ["-q"]) + ["sdist"]
    sandbox.run_setup("setup.py", sdist_args)


def upload_job_to_gcloud(args):
    staging_bucket = "gs://{0}/".format(args.staging_bucket)
    package = search_dir(abspath("dist"), "tsaplay", kind="files", first=True)
    system(
        """gcloud ai-platform jobs submit training {job_name} \\
            --job-dir={job_dir} \\
            --module-name={module_name} \\
            --staging-bucket={staging_bucket} \\
            --packages={package_name} \\
            --config={config_path} \\
            {stream_logs}""".format(
            job_name=args.job_id,
            staging_bucket=staging_bucket,
            job_dir=staging_bucket + (args.job_dir or args.job_id),
            module_name="tsaplay.task",
            package_name=package,
            config_path=abspath(join("gcp", "_config.json")),
            stream_logs="--stream-logs" if args.stream_logs else "",
        )
    )


def submit_job(args):
    clean_dirs(FEATURES_DEST, DATASETS_DEST, EMBEDDINGS_DEST)
    write_gcloud_config(args)
    prepare_job_assets(args)
    upload_job_to_gcloud(args)


def main():
    parser = argument_parser()
    args = parser.parse_args()
    if args.jobs_file:
        for line in open(args.jobs_file, "r"):
            submit_job(parser.parse_args(line.split()))
    else:
        submit_job(args)


if __name__ == "__main__":
    main()
