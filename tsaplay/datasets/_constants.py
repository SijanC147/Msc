from os import getcwd
from os.path import join
import tsaplay.datasets._parsers as PARSERS

PARENT_DIR = join(getcwd(), "tsaplay", "datasets", "data")
DEBUG_PATH = join(PARENT_DIR, "DebugDataset")
DEBUG_PARSER = PARSERS.dong_parser
DONG2014_PATH = join(PARENT_DIR, "Dong2014")
DONG2014_PARSER = PARSERS.dong_parser
NAKOV2016_PATH = join(PARENT_DIR, "Nakov2016")
NAKOV2016_PARSER = PARSERS.nakov_parser
SAEIDI2016_PATH = join(PARENT_DIR, "Saeidi2016")
SAEIDI2016_PARSER = PARSERS.saeidi_parser
WANG2017_PATH = join(PARENT_DIR, "Wang2017")
WANG2017_PARSER = PARSERS.wang_parser
XUE2018_PATH = join(PARENT_DIR, "Xue2018")
XUE2018_RESTAURANTS_PATH = join(XUE2018_PATH, "Restaurants")
XUE2018_LAPTOPS_PATH = join(XUE2018_PATH, "Laptops")
XUE2018_PARSER = PARSERS.xue_parser
ROSENTHAL2015_PATH = join(PARENT_DIR, "Rosenthal2015")
ROSENTHAL2015_PARSER = PARSERS.rosenthal_parser
