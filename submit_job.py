import argparse
from os import makedirs, system
from os.path import join, dirname, abspath
from datetime import datetime
from shutil import copytree, rmtree
from json import dump
from setuptools import sandbox
from tsaplay.task import (
    get_feature_provider,
    argument_parser as task_argument_parser,
)
from tsaplay.utils.decorators import timeit
from tsaplay.utils.io import search_dir, platform, cprnt
import tsaplay.models as tsa_models
from tsaplay.datasets import Dataset
from tsaplay.features import FeatureProvider
from tsaplay.embeddings import (
    Embedding,
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

TEMP_PATH = "temp"
TEMP_FEATURES_PATH = join("tsaplay", "assets", "_features")
TEMP_EMBEDDING_PATH = join("tsaplay", "assets", "_embeddings")
TEMP_DATASET_PATH = join("tsaplay", "assets", "_datasets")

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
        "--task-args",
        "-t",
        help="Arguments to pass forward to the task module",
        nargs=argparse.REMAINDER,
        required=True,
    )

    return parser


def create_temp_folder():
    temp_folder_name = datetime.now().strftime("%Y%m%d-%H%M%S")
    temp_folder = join(TEMP_PATH, temp_folder_name)
    makedirs(temp_folder)
    return temp_folder


def copy_over_dataset_data(datasets, target_temp_path):
    for dataset in datasets:
        name_index = dataset.gen_dir.index(dataset.name)
        path_wrt_name = dataset.gen_dir[name_index:]
        target_path = join(target_temp_path, path_wrt_name)
        copytree(dataset.gen_dir, target_path)


def copy_over_tf_records(feature_provider, embedding_name, target_temp_path):
    train_records = feature_provider.train_tfrecords
    test_records = feature_provider.test_tfrecords
    for (tf_train, tf_test) in zip(train_records, test_records):
        parent_train, parent_test = dirname(tf_train), dirname(tf_test)
        index_train = parent_train.index(embedding_name)
        index_test = parent_test.index(embedding_name)
        target_train = join(target_temp_path, parent_train[index_train:])
        target_test = join(target_temp_path, parent_test[index_test:])
        copytree(parent_train, target_train)
        copytree(parent_test, target_test)


def setup_temp_data(feature_provider):
    embedding = feature_provider.embedding
    datasets = feature_provider.datasets
    temp_folder = create_temp_folder()
    features_temp_path = join(temp_folder, "features")
    embedding_temp_path = join(temp_folder, "embedding", embedding.name)
    datasets_temp_path = join(temp_folder, "datasets")

    copy_over_dataset_data(datasets, datasets_temp_path)
    copy_over_tf_records(feature_provider, embedding.name, features_temp_path)
    copytree(embedding.gen_dir, embedding_temp_path)

    return temp_folder


def copy_to_assets(temp_folder):
    copytree(join(temp_folder, "datasets"), TEMP_DATASET_PATH)
    copytree(join(temp_folder, "features"), TEMP_FEATURES_PATH)
    copytree(join(temp_folder, "embedding"), TEMP_EMBEDDING_PATH)


def clean_prev_input_data():
    rmtree(TEMP_DATASET_PATH, ignore_errors=True)
    rmtree(TEMP_FEATURES_PATH, ignore_errors=True)
    rmtree(TEMP_EMBEDDING_PATH, ignore_errors=True)


def prepare_job_assets(args):
    task_args = task_argument_parser().parse_args(args.task_args)
    feature_provider = get_feature_provider(task_args)
    temp_folder = setup_temp_data(feature_provider)
    clean_prev_input_data()
    copy_to_assets(temp_folder)
    write_gcloud_config(args)
    sandbox.run_setup("setup.py", ["sdist"])


def write_gcloud_config(args):
    job_labels = args.job_labels or {}
    if job_labels:
        job_labels = [label.split("=") for label in job_labels]
        job_labels = {label[0]: label[1] for label in job_labels}

    gcloud_config = {
        "jobId": args.job_id,
        "labels": job_labels,
        "trainingInput": {
            "scaleTier": "CUSTOM",
            "masterType": "n1-highmem-4",
            "workerType": "n1-highmem-4",
            "parameterServerType": "n1-highmem-4",
            "workerCount": 2,
            "parameterServerCount": 2,
            "pythonVersion": "3.5",
            "runtimeVersion": "1.10",
            "region": "europe-west1",
            "args": args.task_args,
        },
    }
    config_file_path = join("gcp", "_config.json")
    with open(config_file_path, "w") as config_file:
        dump(gcloud_config, config_file, indent=4)


@timeit("Uploading job to gcloud...", "Job uploaded")
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


def submit_single_job(args):
    prepare_job_assets(args)
    upload_job_to_gcloud(args)


def submit_multiple_jobs(jobs_file_path):
    parser = argument_parser()
    for line in open(jobs_file_path, "r"):
        submit_single_job(parser.parse_args(line.split()))


def main():
    parser = argument_parser()
    args = parser.parse_args()
    if args.jobs_file:
        submit_multiple_jobs(args.jobs_file)
    else:
        submit_single_job(args)


if __name__ == "__main__":
    main()
