from os import getcwd
from os.path import join
import datasets._parsers as PARSERS

__data = join(getcwd(), "datasets", "data")

DEBUG_PATH = join(__data, "debug_dataset")
DEBUG_PARSER = PARSERS.dong_parser
DONG2014_PATH = join(__data, "Dong2014")
DONG2014_PARSER = PARSERS.dong_parser
NAKOV2016_PATH = join(__data, "Nakov2016")
NAKOV2016_PATH = PARSERS.nakov_parser
SAEIDI2016_PATH = join(__data, "Saeidi2016")
SAEIDI2016_PATH = PARSERS.saeidi_parser
WANG2017_PATH = join(__data, "Wang2017")
WANG2017_PATH = PARSERS.wang_parser
XUE2018_PATH = join(__data, "Xue2018")
XUE2018_PATH = PARSERS.xue_parser
ROSENTHAL2015_PATH = join(__data, "Rosenthal2015")
ROSENTHAL2015_PARSER = PARSERS.rosenthal_parser
