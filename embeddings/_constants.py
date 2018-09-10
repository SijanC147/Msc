from os import getcwd
from os.path import join

__data = join(getcwd(), "embeddings", "data")
__glove = join(__data, "GloVe")

DEBUG = join(__data, "debug_embedding", "glove.twitter.debug.25d.txt")
GLOVE_42B = join(__glove, "glove.42B", "glove.42B.300d.txt")
GLOVE_TWITTER_25D = join(__glove, "glove.twitter.27B", "glove.42B.25d.txt")
GLOVE_TWITTER_50D = join(__glove, "glove.twitter.27B", "glove.42B.50d.txt")
GLOVE_TWITTER_100D = join(__glove, "glove.twitter.27B", "glove.42B.100d.txt")
GLOVE_TWITTER_200D = join(__glove, "glove.twitter.27B", "glove.42B.200d.txt")
