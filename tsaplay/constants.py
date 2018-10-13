from os import getcwd, makedirs, environ
from os.path import join, abspath, dirname
from pathlib import Path

PACKAGE_PATH = dirname(abspath(__file__))

# HOME_PATH = environ.get("TSAPLAY_DATA_PARENT", Path.home())
HOME_PATH = "gs://tsaplay-bucket/tsaplay-data/"
DATA_PATH = join(HOME_PATH, "tsaplay-data")

ASSETS_PATH = join(PACKAGE_PATH, "assets")
DEFAULT_FONT_PATH = join(ASSETS_PATH, "Symbola.ttf")
SPACY_MODEL_PATH = join(ASSETS_PATH, "en_core_web_sm")

DATASET_DATA_PATH = join(DATA_PATH, "datasets")
EMBEDDING_DATA_PATH = join(DATA_PATH, "embeddings")
FEATURES_DATA_PATH = join(DATA_PATH, "features")
EXPERIMENT_DATA_PATH = join(DATA_PATH, "experiments")
EXPORTS_DATA_PATH = join(DATA_PATH, "export")

makedirs(DATA_PATH, exist_ok=True)
makedirs(DATASET_DATA_PATH, exist_ok=True)
makedirs(EMBEDDING_DATA_PATH, exist_ok=True)
makedirs(FEATURES_DATA_PATH, exist_ok=True)
makedirs(EXPERIMENT_DATA_PATH, exist_ok=True)
makedirs(EXPORTS_DATA_PATH, exist_ok=True)
