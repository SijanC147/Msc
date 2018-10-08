from os.path import normpath, basename, join, exists
from functools import wraps
from tsaplay.utils.nlp import corpus_from_docs
from tsaplay.utils.io import (
    search_dir,
    corpus_from_csv,
    corpus_to_csv,
    unpickle_file as _unpickle,
    pickle_file as _pickle,
)
from tsaplay.utils.decorators import timeit
import tsaplay.datasets.constants as DATASETS


class Dataset:
    def __init__(self, path, parser):
        self.__parser = self._wrap_parser(parser)
        self.__path = path
        self._initialize_all_internals()

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
    def corpus(self):
        return [*self.__corpus]

    @property
    def all_docs(self):
        return self.__all_docs

    @timeit("Generating corpus for dataset", "Corpus generated")
    def _generate_corpus(self):
        corpus_file = join(self.path, "_corpus.csv")
        if exists(corpus_file):
            corpus = corpus_from_csv(path=corpus_file)
        else:
            corpus = corpus_from_docs(docs=self.all_docs)
            corpus_to_csv(corpus_file, corpus)
        return corpus

    @timeit("Initializing dataset internals", "Dataset internals initialized")
    def _initialize_all_internals(self):
        self.__train_file = search_dir(
            dir=self.__path, query="train", first=True, files_only=True
        )
        self.__test_file = search_dir(
            dir=self.__path, query="test", first=True, files_only=True
        )
        self.__train_dict = self._load_dataset_dictionary(dict_type="train")
        self.__test_dict = self._load_dataset_dictionary(dict_type="test")
        self.__all_docs = set(
            self.__train_dict["sentences"] + self.__test_dict["sentences"]
        )
        self.__corpus = self._generate_corpus()

    @timeit("Loading dataset dictionary", "Dataset dictionary loaded")
    def _load_dataset_dictionary(self, dict_type):
        if dict_type == "train":
            dict_file = join(self.path, "_train_dict.pkl")
            data_file = self.__train_file
        else:
            dict_file = join(self.path, "_test_dict.pkl")
            data_file = self.__test_file
        if exists(dict_file):
            dictionary = _unpickle(path=dict_file)
        else:
            dictionary = self.parser(data_file)
            _pickle(path=dict_file, data=dictionary)
        return dictionary

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
                sentences, targets, labels = _parser(path)
                offsets = [
                    sentence.lower().find(target.lower())
                    for sentence, target in zip(sentences, targets)
                ]
                return {
                    "sentences": sentences,
                    "targets": targets,
                    "offsets": offsets,
                    "labels": labels,
                }

        return wrapper
