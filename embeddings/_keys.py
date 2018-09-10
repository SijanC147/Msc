from os import getcwd
from os.path import join
from utils import ConstantDict

__data_path = join(getcwd(), "embedings", "data")
__glove_path = join(__data_path, "GloVe")

GLOVE = ConstantDict(
    {
        "42B": join(__glove_path, "glove.42B", "glove.42B.300d.txt"),
        "TWITTER": ConstantDict(
            {
                "25D": join(
                    __glove_path, "glove.twitter.27B", "glove.42B.25d.txt"
                ),
                "50D": join(
                    __glove_path, "glove.twitter.27B", "glove.42B.50d.txt"
                ),
                "100D": join(
                    __glove_path, "glove.twitter.27B", "glove.42B.100d.txt"
                ),
                "200D": join(
                    __glove_path, "glove.twitter.27B", "glove.42B.200d.txt"
                ),
            }
        ),
    }
)

DEBUG = ConstantDict(
    {"25D": join(__data_path, "DEBUG", "glove.twitter.debug.25d.txt")}
)
