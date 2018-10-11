from os import makedirs
from os.path import normpath, basename, join, exists
from functools import wraps
import json
import numpy as np
import spacy
from spacy.attrs import ORTH  # pylint: disable=E0611
from tsaplay.features.FeatureProvider import FeatureProvider
from tsaplay.utils.data import resample_data_dict, get_class_distribution
from tsaplay.utils.io import (
    search_dir,
    corpus_from_csv,
    corpus_to_csv,
    unpickle_file as _unpickle,
    pickle_file as _pickle,
)
from tsaplay.utils.decorators import timeit


class Dataset:
    def __init__(self, path, parser, distribution=None):
        self.__parser = self._wrap_parser(parser)
        self.__path = path
        self._initialize_all_internals()
        if distribution is not None:
            self._redistribute_data(distribution)
        else:
            _, _, train_dist = get_class_distribution(self.train_dict)
            _, _, test_dist = get_class_distribution(self.train_dict)
            self.__train_dist_key = "_".join(map(str, train_dist))
            self.__test_dist_key = "_".join(map(str, test_dist))

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
    def train_dist_key(self):
        return self.__train_dist_key

    @property
    def test_dist_key(self):
        return self.__test_dist_key

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

    @property
    def default_classes(self):
        return self.__default_classes

    def get_dist_key(self, mode=None):
        train_dist_key = self.__train_dist_key
        test_dist_key = self.__test_dist_key
        if mode == "train":
            return train_dist_key
        if mode == "test":
            return test_dist_key
        if mode is None:
            if train_dist_key == test_dist_key:
                return train_dist_key
            return "-".join([train_dist_key, test_dist_key])

    @classmethod
    def get_stats_dict(cls, classes=None, **data_dicts):
        stats = {}
        for key, value in data_dicts.items():
            stats[key] = stats.get(key, {})
            dist_data = get_class_distribution(value, all_classes=classes)
            for (_class, count, dist) in zip(*dist_data):
                stats[key].update(
                    {str(_class): {"count": str(count), "percent": str(dist)}}
                )
        return stats

    @classmethod
    @timeit("Generating corpus for dataset", "Corpus generated")
    def generate_corpus(cls, docs, path):
        corpus_file = join(path, "_corpus.csv")
        if exists(corpus_file):
            corpus = corpus_from_csv(path=corpus_file)
        else:
            corpus = cls.corpus_from_docs(docs)
            corpus_to_csv(corpus_file, corpus)
        return corpus

    @classmethod
    def corpus_from_docs(cls, docs):
        corpus = {}

        nlp = spacy.load("en", disable=["parser", "ner"])
        docs_joined = " ".join(map(lambda document: document.strip(), docs))
        if len(docs_joined) > 1000000:
            nlp.max_length = len(docs_joined) + 1
        tokens = nlp(docs_joined)
        tokens = list(filter(FeatureProvider.token_filter, tokens))
        doc = nlp(" ".join(map(lambda token: token.text, tokens)))
        counts = doc.count_by(ORTH)
        words = counts.items()
        for word_id, cnt in sorted(
            words, reverse=True, key=lambda item: item[1]
        ):
            corpus[nlp.vocab.strings[word_id]] = cnt

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
        self.__default_classes = np.unique(
            [label for label in self.__train_dict["labels"]]
        ).tolist()
        self.write_stats_json(
            self.path,
            classes=self.__default_classes,
            train=self.__train_dict,
            test=self.__test_dict,
        )
        self.__all_docs = set(
            self.__train_dict["sentences"] + self.__test_dict["sentences"]
        )
        self.__corpus = self.generate_corpus(self.__all_docs, self.path)

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

    @classmethod
    @timeit("Exporting dataset stats", "Dataset stats exported")
    def write_stats_json(cls, path, classes=None, **data_dicts):
        stats = cls.get_stats_dict(classes=classes, **data_dicts)
        stats_file_path = join(path, "_stats.json")
        if not (exists(stats_file_path)):
            with open(stats_file_path, "w+") as stats_file:
                json.dump(stats, stats_file, indent=4)

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
            except ValueError:
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

    @timeit("Redistributing dataset", "Dataset redistributed")
    def _redistribute_data(self, distribution):
        self.__train_dist_key = None
        self.__test_dist_key = None
        dists_dir = join(self.path, "_dists")
        makedirs(dists_dir, exist_ok=True)
        if isinstance(distribution, list):
            dist_list = distribution
            distribution = {"train": dist_list, "test": dist_list}
        elif not isinstance(distribution, dict):
            raise ValueError
        for key, dist_values in distribution.items():
            dist_folder = "_".join([str(int(v * 100)) for v in dist_values])
            dist_path = join(dists_dir, dist_folder)
            makedirs(dist_path, exist_ok=True)
            dist_file = join(dist_path, "_" + key + "_dict.pkl")
            if exists(dist_file):
                resampled_dict = _unpickle(path=dist_file)
                if key == "train":
                    self.__train_dict = resampled_dict
                    self.__train_dist_key = dist_folder
                else:
                    self.__test_dict = resampled_dict
                    self.__test_dist_key = dist_folder
            else:
                all_docs = []
                data_dicts = {}
                for mode in ["train", "test"]:
                    dist_file = join(dist_path, "_" + mode + "_dict.pkl")
                    if mode == "train":
                        orig_dict = self.__train_dict
                    else:
                        orig_dict = self.__test_dict

                    resampled_dict = resample_data_dict(orig_dict, dist_values)
                    _pickle(path=dist_file, data=resampled_dict)
                    data_dicts[mode] = resampled_dict
                    all_docs += resampled_dict["sentences"]

                self.write_stats_json(
                    dist_path, classes=self.default_classes, **data_dicts
                )
                self.generate_corpus(set(all_docs), dist_path)

                if key == "train":
                    self.__train_dict = data_dicts[key]
                    self.__train_dist_key = dist_folder
                elif key == "test":
                    self.__test_dict = data_dicts[key]
                    self.__test_dist_key = dist_folder

        self.__all_docs = set(
            self.__train_dict["sentences"] + self.__test_dict["sentences"]
        )
        self.__corpus = self.generate_corpus(self.__all_docs, dist_path)
