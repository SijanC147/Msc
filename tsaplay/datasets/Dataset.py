from time import time as _time
from os import listdir, makedirs
from os.path import normpath, basename, isfile, join, exists, dirname
from statistics import mean
from functools import wraps
from csv import DictWriter, DictReader
from tsaplay.utils._nlp import (
    token_filter,
    re_dist,
    inspect_dist,
    get_sentence_contexts,
    corpus_from_docs,
    get_sentence_target_features,
)
from tsaplay.utils._io import (
    search_dir,
    corpus_from_csv,
    corpus_to_csv,
    write_embedding_to_disk,
    write_emb_tsv_to_disk,
    unpickle_file as _unpickle,
    pickle_file as _pickle,
)
import tsaplay.datasets._constants as DATASETS


class Dataset:
    def __init__(self, path, parser):
        self.__parser = self._wrap_parser(parser)
        self.path = path

    @property
    def name(self):
        return basename(normpath(self.path))

    @property
    def path(self):
        return self.__path

    @property
    def parser(self):
        return self.__parser

    @property
    def train_file(self):
        return self.__train_file

    @property
    def test_file(self):
        return self.__test_file

    @property
    def train_dict(self):
        return self.__train_dict

    @property
    def test_dict(self):
        return self.__test_dict

    @property
    def gen_dir(self):
        return self.__gen_dir

    @property
    def corpus(self):
        return self.__corpus

    @property
    def all_docs(self):
        return self.__all_docs

    @path.setter
    def path(self, path):
        try:
            path_changed = self.path != path
        except:
            path_changed = True

        if path_changed:
            self.__path = path
            self._reset()

    def _reset(self):
        self.__gen_dir = join(self.path, "_generated")
        makedirs(self.__gen_dir, exist_ok=True)
        self.__train_file = search_dir(
            dir=self.__path, query="train", first=True, files_only=True
        )
        self.__test_file = search_dir(
            dir=self.__path, query="test", first=True, files_only=True
        )
        self.__train_dict = self._load_dataset_dictionary(dict_type="trian")
        self.__test_dict = self._load_dataset_dictionary(dict_type="test")
        self.__all_docs = set(
            self.__train_dict["sentences"] + self.__test_dict["sentences"]
        )
        self.__corpus = self._generate_corpus()

    def _load_dataset_dictionary(self, dict_type):
        if dict_type == "train":
            dict_file = join(self.gen_dir, "train_dict.pkl")
            data_file = self.__train_file
        else:
            dict_file = join(self.gen_dir, "test_dict.pkl")
            data_file = self.__test_file
        if exists(dict_file):
            dictionary = _unpickle(path=dict_file)
        else:
            dictionary = self.parser(data_file)
            _pickle(path=dict_file, data=dictionary)
        return dictionary

    def _generate_corpus(self):
        corpus_file = join(self.gen_dir, "corpus.csv")
        if exists(corpus_file):
            corpus = corpus_from_csv(path=corpus_file)
        else:
            corpus = corpus_from_docs(docs=self.all_docs)
            corpus_to_csv(corpus_file, corpus)
        return corpus

    def _wrap_parser(self, _parser):
        @wraps(_parser)
        def wrapper(path):
            try:
                sentences, targets, offsets, labels = _parser(path)
                return {
                    "sentences": sentences,
                    "targets": targets,
                    "offsets": offsets,
                    "labels": labels,
                }
            except:
                sentences, targets, labels, = _parser(path)
                return {
                    "sentences": sentences,
                    "targets": targets,
                    "labels": labels,
                    "offsets": [None] * len(labels),
                }

        return wrapper
