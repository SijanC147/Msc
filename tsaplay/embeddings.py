from os.path import join, exists
import numpy as np
import gensim.downloader as gensim_data
from gensim.models import KeyedVectors
from tsaplay.constants import (
    EMBEDDING_DATA_PATH,
    PAD_TOKEN,
    EMBEDDING_SHORTHANDS,
)
from tsaplay.utils.io import list_folders
from tsaplay.utils.data import hash_data


class Embedding:
    def __init__(self, name, filters=None):
        self._name = None
        self._uid = None
        self._gen_dir = None
        self._gensim_model = None
        self._vocab = None
        self._vocab_size = None
        self._dim_size = None
        self._vectors = None

        self._init_gen_dir(name)
        self._init_gensim_model(self.name, self.gen_dir)

        self._uid = "{name}{filter_hash}".format(
            name=self._name, filter_hash=hash_data(filters)
        )

    @property
    def name(self):
        return self._name

    @property
    def uid(self):
        return self._uid

    @property
    def gen_dir(self):
        return self._gen_dir

    @property
    def vocab(self):
        return self._vocab

    @property
    def vocab_size(self):
        return self._vocab_size

    @property
    def dim_size(self):
        return self._dim_size

    @property
    def vectors(self):
        return self._vectors

    def _init_gen_dir(self, name):
        name = EMBEDDING_SHORTHANDS.get(name, name)
        data_root = EMBEDDING_DATA_PATH
        gen_dir = join(data_root, name)
        gensim_models = gensim_data.info(name_only=True)["models"]
        if not exists(gen_dir) and name not in gensim_models:
            offline_models = list_folders(data_root)
            shorthand_names = [*EMBEDDING_SHORTHANDS]
            available_embeddings = set(
                offline_models + gensim_models + shorthand_names
            )
            raise ValueError(
                """Expected Embedding name to be one of {0},\
                got {1}.""".format(
                    available_embeddings, name
                )
            )
        else:
            self._gen_dir = gen_dir
            self._name = name

    def _init_gensim_model(self, name, path):
        gensim_model_path = join(path, "_gensim_model.bin")
        if exists(gensim_model_path):
            self._gensim_model = KeyedVectors.load(gensim_model_path)
        else:
            self._gensim_model = gensim_data.load(name)
            self._gensim_model.save(gensim_model_path)
        self._dim_size = self._gensim_model.vector_size
        self._vocab = [PAD_TOKEN] + self._gensim_model.index2word
        self._vocab_size = len(self._vocab)
        pad_value = [np.zeros(shape=self._dim_size).astype(np.float32)]
        self._vectors = np.concatenate([pad_value, self._gensim_model.vectors])
