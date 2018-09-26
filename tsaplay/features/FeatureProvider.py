import tensorflow as tf
from os.path import join
from os import getcwd, makedirs

FEATURE_DATA_PATH = join(getcwd(), "features", "data")


class FeatureProvider:
    def __init__(self, dataset, embedding):
        self.__dataset = dataset
        self.__embedding = embedding
        makedirs(self.gen_dir, exist_ok=True)

    @property
    def gen_dir(self):
        return join(
            FEATURE_DATA_PATH, self.__embedding.name, self.__dataset.name
        )
