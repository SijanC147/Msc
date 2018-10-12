from os import getcwd
from os.path import join, abspath, dirname
from pathlib import Path
from collections import namedtuple

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


PACKAGE_PATH = dirname(abspath(__file__))
ASSETS_PATH = join(PACKAGE_PATH, "assets")

DEFAULT_FONT_PATH = join(ASSETS_PATH, "Symbola.ttf")

DATA_PATH = join(Path.home(), ".tsaplay-data")

DATASET_DATA_PATH = join(DATA_PATH, "datasets")
EMBEDDING_DATA_PATH = join(DATA_PATH, "embeddings")
FEATURES_DATA_PATH = join(DATA_PATH, "features")

EXPERIMENT_DATA_PATH = join(DATA_PATH, "experiments")
EXPORT_MODEL_PATH = join(DATA_PATH, "export")
