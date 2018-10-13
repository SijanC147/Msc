from os import getcwd, makedirs
from os.path import join
from pathlib import Path
import pkg_resources as pkg

DEFAULT_FONT = pkg.resource_filename(__name__, "assets/Symbola.ttf")
SPACY_MODEL = pkg.resource_filename(__name__, "assets/en_core_web_sm")

DEBUG_ASSETS = pkg.resource_filename(__name__, "assets/DebugDataset")
DEBUGV2_ASSETS = pkg.resource_filename(__name__, "assets/DebugDatasetV2")
DONG_ASSETS = pkg.resource_filename(__name__, "assets/Dong2014")
LAPTOPS_ASSETS = pkg.resource_filename(__name__, "assets/Laptops")
NAKOV_ASSETS = pkg.resource_filename(__name__, "assets/Nakov2016")
RESTAURANTS_ASSETS = pkg.resource_filename(__name__, "assets/Restaurants")
ROSENTHAL_ASSETS = pkg.resource_filename(__name__, "assets/Rosenthal2015")
SAEIDI_ASSETS = pkg.resource_filename(__name__, "assets/Saeidi2016")
WANG_ASSETS = pkg.resource_filename(__name__, "assets/Wang2017")
XUE_ASSETS = pkg.resource_filename(__name__, "assets/Xue2018")

DATA_PATH = join(Path.home(), "tsaplay-data")
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
