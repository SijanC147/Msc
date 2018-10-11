from os import getcwd
from os.path import join
from collections import namedtuple
import tsaplay.datasets.parsers as dataset_parsers


DatasetKey = namedtuple("DatasetKey", ["path", "parser"])

DATASET_DATA_PATH = join(getcwd(), "tsaplay", "datasets", "data")
EMBEDDING_DATA_PATH = join(getcwd(), "tsaplay", "embeddings", "data")
EXPERIMENT_DATA_PATH = join(getcwd(), "tsaplay", "experiments", "data")
FEATURES_DATA_PATH = join(getcwd(), "tsaplay", "features", "data")
EXPORT_MODEL_PATH = join(getcwd(), "export")

FASTTEXT_WIKI_300 = "fasttext-wiki-news-subwords-300"
GLOVE_TWITTER_25 = "glove-twitter-25"
GLOVE_TWITTER_50 = "glove-twitter-50"
GLOVE_TWITTER_100 = "glove-twitter-100"
GLOVE_TWITTER_200 = "glove-twitter-200"
GLOVE_WIKI_GIGA_50 = "glove-wiki-gigaword-50"
GLOVE_WIKI_GIGA_100 = "glove-wiki-gigaword-100"
GLOVE_WIKI_GIGA_200 = "glove-wiki-gigaword-200"
GLOVE_WIKI_GIGA_300 = "glove-wiki-gigaword-300"
GLOVE_COMMON42_300 = "glove-cc42-300"
GLOVE_COMMON840_300 = "glove-cc840-300"
W2V_GOOGLE_300 = "word2vec-google-news-300"
W2V_RUS_300 = "word2vec-ruscorpora-300"

DEBUG_PATH = join(DATASET_DATA_PATH, "DebugDataset")
DEBUGV2_PATH = join(DATASET_DATA_PATH, "DebugDatasetV2")
DONG2014_PATH = join(DATASET_DATA_PATH, "Dong2014")
NAKOV2016_PATH = join(DATASET_DATA_PATH, "Nakov2016")
SAEIDI2016_PATH = join(DATASET_DATA_PATH, "Saeidi2016")
WANG2017_PATH = join(DATASET_DATA_PATH, "Wang2017")
XUE2018_PATH = join(DATASET_DATA_PATH, "Xue2018")
XUE2018_RESTAURANTS_PATH = join(XUE2018_PATH, "Restaurants")
XUE2018_LAPTOPS_PATH = join(XUE2018_PATH, "Laptops")
ROSENTHAL2015_PATH = join(DATASET_DATA_PATH, "Rosenthal2015")

DEBUG_PARSER = dataset_parsers.dong_parser
DEBUGV2_PARSER = dataset_parsers.dong_parser
DONG2014_PARSER = dataset_parsers.dong_parser
NAKOV2016_PARSER = dataset_parsers.nakov_parser
SAEIDI2016_PARSER = dataset_parsers.saeidi_parser
WANG2017_PARSER = dataset_parsers.wang_parser
XUE2018_PARSER = dataset_parsers.xue_parser
ROSENTHAL2015_PARSER = dataset_parsers.rosenthal_parser

DEBUG = DatasetKey(DEBUG_PATH, DEBUG_PARSER)
DEBUGV2 = DatasetKey(DEBUGV2_PATH, DEBUGV2_PARSER)
DONG = DatasetKey(DONG2014_PATH, DONG2014_PARSER)
NAKOV = DatasetKey(NAKOV2016_PATH, NAKOV2016_PARSER)
SAEIDI = DatasetKey(SAEIDI2016_PATH, SAEIDI2016_PARSER)
WANG = DatasetKey(WANG2017_PATH, WANG2017_PARSER)
XUE = DatasetKey(XUE2018_PATH, XUE2018_PARSER)
RESTAURANTS = DatasetKey(XUE2018_RESTAURANTS_PATH, XUE2018_PARSER)
LAPTOPS = DatasetKey(XUE2018_LAPTOPS_PATH, XUE2018_PARSER)
ROSENTHAL = DatasetKey(ROSENTHAL2015_PATH, ROSENTHAL2015_PARSER)
