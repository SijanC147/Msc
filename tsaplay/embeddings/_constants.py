from os import getcwd
from os.path import join

__data = join(getcwd(), "tsaplay", "embeddings", "data")
__glove = join(__data, "GloVe")
__debug = join(__data, "debug_embedding")

DEBUG = join(__debug, "glove.twitter.debug.25d.txt")
GLOVE_42B = join(__glove, "glove.42B.300d.txt")
GLOVE_TWITTER_25D = join(__glove, "glove.twitter.27B.25d.txt")
GLOVE_TWITTER_50D = join(__glove, "glove.twitter.27B.50d.txt")
GLOVE_TWITTER_100D = join(__glove, "glove.twitter.27B.100d.txt")
GLOVE_TWITTER_200D = join(__glove, "glove.twitter.27B.200d.txt")
