import argparse
from os import makedirs, rename, listdir
from os.path import join, dirname, exists
from datetime import datetime
from shutil import copytree, rmtree
import tsaplay.models as tsa_models
from tsaplay.datasets import Dataset
from tsaplay.embeddings import Embedding
from tsaplay.features import FeatureProvider

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
    "tdlstm": tsa_models.TdLstm,
    "tclstm": tsa_models.TcLstm,
    "lcrrot": tsa_models.LcrRot,
    "ian": tsa_models.Ian,
    "memnet": tsa_models.MemNet,
    "ram": tsa_models.Ram,
}

TEMP_PATH = "temp"
TEMP_INPUTDATA_PATH = join("tsaplay", "assets", "_inputdata")
TEMP_EMBEDDING_PATH = join("tsaplay", "assets", "_embedding")
TEMP_DATASET_PATH = join("tsaplay", "assets", "_datasets")


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
        target_train = join(target_temp_path + parent_train[index_train:])
        target_test = join(target_temp_path + parent_test[index_test:])
        copytree(parent_train, target_train)
        copytree(parent_test, target_test)


def setup_temp_data(embedding, datasets):
    embedding = Embedding(EMBEDDINGS.get(embedding))
    datasets = [Dataset(*DATASETS.get(dataset)) for dataset in datasets]
    feature_provider = FeatureProvider(datasets, embedding)

    temp_folder = create_temp_folder()
    inputdata_temp_path = join(temp_folder, "inputdata")
    embedding_temp_path = join(temp_folder, "embedding", embedding.name)
    datasets_temp_path = join(temp_folder, "datasets")

    copy_over_dataset_data(datasets, datasets_temp_path)
    copy_over_tf_records(feature_provider, embedding.name, inputdata_temp_path)
    copytree(embedding.gen_dir, embedding_temp_path)

    return temp_folder


def copy_to_assets(temp_folder):
    copytree(join(temp_folder, "datasets"), TEMP_DATASET_PATH)
    copytree(join(temp_folder, "inputdata"), TEMP_INPUTDATA_PATH)
    copytree(join(temp_folder, "embedding"), TEMP_EMBEDDING_PATH)


def clean_prev_input_data():
    if exists(TEMP_INPUTDATA_PATH):
        rmtree(TEMP_INPUTDATA_PATH)


def prepare_assets():
    args = get_arguments()
    temp_folder = setup_temp_data(args.embedding, args.datasets)
    clean_prev_input_data()
    copy_to_assets(temp_folder)


if __name__ == "__main__":
    prepare_assets()
