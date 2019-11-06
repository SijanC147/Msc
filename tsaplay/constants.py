from os import makedirs
from os.path import join
from functools import partial
from pathlib import Path
import pkg_resources as pkg
import numpy as np

# TF_RANDOM_SEED = None
TF_RANDOM_SEED = 1234
NP_RANDOM_SEED = 1234
SAVE_SUMMARY_STEPS = 100
SAVE_CHECKPOINTS_STEPS = 1000
LOG_STEP_COUNT_STEPS = 1000
KEEP_CHECKPOINT_MAX = 5
DELIMITER = "<SEP>"
TF_DELIMITER = " "
PAD_TOKEN = "<PAD>"
BUCKET_TOKEN = "<BKT-{num}>"
TF_RECORD_SHARDS = 10
MAX_EMBEDDING_SHARDS = 1
DEFAULT_COMET_WORKSPACE = "msc"
DEFAULT_OOV_FN = partial(np.random.uniform, low=-0.1, high=0.1)
MODELS_PATH = pkg.resource_filename(__name__, "models")
ASSETS_PATH = pkg.resource_filename(__name__, "assets")
SPACY_MODEL = join(ASSETS_PATH, "en_core_web_sm")
DEFAULT_FONT = pkg.resource_filename(__name__, "fonts/Symbola.ttf")

EMBEDDING_SHORTHANDS = {
    "fasttext": "fasttext-wiki-news-subwords-300",
    "twitter-25": "glove-twitter-25",
    "twitter-50": "glove-twitter-50",
    "twitter-100": "glove-twitter-100",
    "twitter-200": "glove-twitter-200",
    "wiki-50": "glove-wiki-gigaword-50",
    "wiki-100": "glove-wiki-gigaword-100",
    "wiki-200": "glove-wiki-gigaword-200",
    "wiki-300": "glove-wiki-gigaword-300",
    "commoncrawl-42": "glove-cc42-300",
    "commoncrawl-840": "glove-cc840-300",
    "w2v-google-300": "word2vec-google-news-300",
    "w2v-rus-300": "word2vec-ruscorpora-300",
}

try:
    DATA_PATH = join(Path.home(), "tsaplay-data")
except TypeError:
    DATA_PATH = ASSETS_PATH

DATASET_DATA_PATH = join(DATA_PATH, "_datasets")
EMBEDDING_DATA_PATH = join(DATA_PATH, "_embeddings")
FEATURES_DATA_PATH = join(DATA_PATH, "_features")
EXPERIMENT_DATA_PATH = join(DATA_PATH, "experiments")
EXPORTS_DATA_PATH = join(DATA_PATH, "export")

makedirs(DATA_PATH, exist_ok=True)
makedirs(DATASET_DATA_PATH, exist_ok=True)
makedirs(EMBEDDING_DATA_PATH, exist_ok=True)
makedirs(FEATURES_DATA_PATH, exist_ok=True)
makedirs(EXPERIMENT_DATA_PATH, exist_ok=True)
makedirs(EXPORTS_DATA_PATH, exist_ok=True)
