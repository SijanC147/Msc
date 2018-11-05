import argparse
from os import makedirs, system
from os.path import join, dirname, abspath
from json import dump
from setuptools import sandbox
from tsaplay.task import (
    get_feature_provider,
    argument_parser as task_argument_parser,
    args_to_dict,
)
from tsaplay.utils.io import search_dir, copy
from tsaplay.constants import (
    ASSETS_PATH,
    DATASET_DATA_PATH,
    EMBEDDING_DATA_PATH,
    FEATURES_DATA_PATH,
)
import tsaplay.models as tsa_models

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
        "--stream-logs",
        "-stream",
        help="Include flag to stream logs in gcloud submit command",
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


def prepare_job_assets(args):
    task_args = task_argument_parser().parse_args(args.task_args)
    feature_provider = get_feature_provider(
        task_args.datasets,
        task_args.embedding,
        args_to_dict(task_args.model_params),
    )
    asset_dst_path = join(ASSETS_PATH, "_{}")
    features_dst = join(
        asset_dst_path.format("features"), feature_provider.uid
    )
    makedirs(features_dst, exist_ok=True)
    for dataset in feature_provider.datasets:
        copy(
            dataset.gen_dir,
            asset_dst_path.format("datasets"),
            rel=DATASET_DATA_PATH,
        )
    copy(
        feature_provider.embedding.gen_dir,
        asset_dst_path.format("embeddings"),
        rel=EMBEDDING_DATA_PATH,
    )
    copy(
        feature_provider.embedding_params["_vocab_file"],
        features_dst,
        rel=FEATURES_DATA_PATH,
    )
    for mode in ["train", "test"]:
        tfrecord_src = dirname(
            getattr(feature_provider, "{}_tfrecords".format(mode))
        )
        tokens_src = join(
            feature_provider.gen_dir, "_{}_tokens.pkl".format(mode)
        )
        copy(tfrecord_src, features_dst, rel=FEATURES_DATA_PATH)
        copy(tokens_src, features_dst, rel=FEATURES_DATA_PATH)

    write_gcloud_config(args)
    sdist_args = ([] if args.show_sdist else ["-q"]) + ["sdist"]
    sandbox.run_setup("setup.py", sdist_args)


def write_gcloud_config(args):
    gcloud_config = {
        "jobId": args.job_id,
        "labels": args_to_dict(args.job_labels),
        "trainingInput": {
            "scaleTier": "CUSTOM",
            "masterType": "standard_gpu",
            # "workerType": "n1-standard-16",
            # "parameterServerType": "standard_gpu",
            # "workerCount": 5,
            # "parameterServerCount": 2,
            "pythonVersion": "3.5",
            "runtimeVersion": "1.10",
            "region": "europe-west1",
            "args": args.task_args,
        },
    }
    config_file_path = join("gcp", "_config.json")
    with open(config_file_path, "w") as config_file:
        dump(gcloud_config, config_file, indent=4)


def upload_job_to_gcloud(args):
    staging_bucket = "gs://tsaplay-bucket/"
    package = search_dir(abspath("dist"), "tsaplay", kind="files", first=True)
    system(
        """gcloud ml-engine jobs submit training {job_name} \\
--job-dir={job_dir} \\
--module-name={module_name} \\
--staging-bucket={staging_bucket} \\
--packages={package_name} \\
--config={config_path} {stream_logs}""".format(
            job_name=args.job_id,
            staging_bucket=staging_bucket,
            job_dir=staging_bucket + (args.job_dir or args.job_id),
            module_name="tsaplay.task",
            package_name=package,
            config_path=abspath(join("gcp", "_config.json")),
            stream_logs="--stream_logs" if args.stream_logs else "",
        )
    )


def submit_job(args):
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
