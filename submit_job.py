import argparse
from os import makedirs, system
from os.path import join, dirname, abspath
from datetime import datetime
from shutil import copytree, rmtree
from json import dump
from setuptools import sandbox
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


def get_arguments():
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
        "--job-id", "-jid", type=str, help="ID of the job being submitted"
    )

    parser.add_argument(
        "--job-dir",
        help="GCS location to write checkpoints to and export models",
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

    return parser.parse_args()


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


def setup_temp_data(embedding, datasets):
    embedding = Embedding(EMBEDDINGS.get(embedding))
    datasets = [Dataset(dataset_name) for dataset_name in datasets]
    feature_provider = FeatureProvider(datasets, embedding)

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


def prepare_assets(args):
    temp_folder = setup_temp_data(args.embedding, args.datasets)
    clean_prev_input_data()
    copy_to_assets(temp_folder)
    write_gcloud_config(args)


def write_gcloud_config(args):
    args_dict = vars(args)
    args_list = []
    for (key, value) in args_dict.items():
        if not value or key == "job_id":
            continue
        if isinstance(value, list):
            value = " ".join(map(str, value))
        else:
            value = str(value)
        args_list.append("--" + key.replace("_", "-") + "=" + value)
    gcloud_config = {
        "jobId": "my_job",
        "labels": {"type": "dev", "owner": "sean"},
        "trainingInput": {
            "scaleTier": "CUSTOM",
            "masterType": "standard_gpu",
            "workerType": "standard_gpu",
            "parameterServerType": "standard_gpu",
            "workerCount": 2,
            "parameterServerCount": 6,
            "pythonVersion": "3.5",
            "runtimeVersion": "1.10",
            "region": "europe-west1",
            "args": args_list,
        },
    }
    config_file_path = join("gcp", "_config.json")
    with open(config_file_path, "w") as config_file:
        dump(gcloud_config, config_file, indent=4)


def write_gcloud_cmd_script(args):
    gcloud_cmd = """gcloud ml-engine jobs submit training {job_name} \\
--job-dir={job_dir} \\
--module-name={module_name} \\
--staging-bucket={staging_bucket} \\
--packages={package_name} \\
--config={config_path} \\
--stream-logs""".format(
        job_name=args.job_id,
        job_dir="gs://tsaplay-bucket/{}".format(args.job_id),
        module_name="tsaplay.task",
        staging_bucket="gs://tsaplay-bucket/",
        package_name=abspath(
            search_dir(join("dist"), query="tsaplay", kind="files", first=True)
        ),
        config_path=abspath(join("gcp", "_config.json")),
    )

    cmd_script = """from os import system\n
\nsystem(\"\"\"{0}\"\"\")\n""".format(
        gcloud_cmd
    )

    cmd_file_path = abspath(join("gcp", "_cmd.py"))
    with open(cmd_file_path, "w") as cmd_file:
        cmd_file.write(cmd_script)

    submit_job_cmd = "pyenv shell 2.7.15 && pyenv exec python {} && pyenv shell -".format(
        cmd_file_path
    )

    print("Execute following command to submit job: ")
    cprnt(submit_job_cmd)

    if platform() == "MacOS":
        system('echo "{}" | pbcopy'.format(submit_job_cmd))
        cprnt(bow="Copied to clipboard.")


def main(args):
    prepare_assets(args)
    sandbox.run_setup("setup.py", ["sdist"])
    write_gcloud_cmd_script(args)


if __name__ == "__main__":
    main(get_arguments())
