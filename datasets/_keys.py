from os import getcwd
from os.path import join
from utils import ConstantDict
import datasets._parsers as PARSERS

__data_path = join(getcwd(), "datasets", "data")

DONG2014 = ConstantDict(
    {"PATH": join(__data_path, "Dong2014"), "PARSER": PARSERS.dong_parser}
)
NAKOV2016 = ConstantDict(
    {"PATH": join(__data_path, "Nakov2016"), "PARSER": PARSERS.nakov_parser}
)
SAEIDI2016 = ConstantDict(
    {"PATH": join(__data_path, "Saeidi2016"), "PARSER": PARSERS.saeidi_parser}
)
WANG2017 = ConstantDict(
    {"PATH": join(__data_path, "Wang2017"), "PARSER": PARSERS.wang_parser}
)
XUE2018 = ConstantDict(
    {"PATH": join(__data_path, "Xue2018"), "PARSER": PARSERS.xue_parser}
)
ROSENTHAL2015 = ConstantDict(
    {
        "PATH": join(__data_path, "Rosenthal2015"),
        "PARSER": PARSERS.rosenthal_parser,
    }
)
