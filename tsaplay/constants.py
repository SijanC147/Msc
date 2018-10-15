from os import getcwd, makedirs
from os.path import join
from pathlib import Path
import pkg_resources as pkg

SPACY_MODEL = pkg.resource_filename(__name__, "assets/en_core_web_sm")
DEFAULT_FONT = pkg.resource_filename(__name__, "fonts/Symbola.ttf")

try:
    DATA_PATH = join(Path.home(), "tsaplay-data")
except TypeError:
    DATA_PATH = pkg.resource_filename(__name__, "assets")

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
