from os import makedirs
from os.path import normpath, basename, join, exists
from functools import wraps
import json
import numpy as np
from tsaplay.utils.nlp import corpus_from_docs
from tsaplay.utils.data import resample_data_dict
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
    def __init__(self, path, parser, distribution=None):
        self.__parser = self._wrap_parser(parser)
        self.__path = path
        self._initialize_all_internals()
        if distribution is not None:
            self._redistribute_data(distribution)
        else:
            self.__train_dist_key = self._set_dist_key(self.train_dict)
            self.__test_dist_key = self._set_dist_key(self.test_dict)

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

    def get_dist_key(self, mode=None):
        # if self.__train_dist_key is None and self.__test_dist_key is None:
        #     return None
        # default_dist_key = "-DEFAULT-"
        # train_dist_key = self.__train_dist_key or default_dist_key
        # test_dist_key = self.__test_dist_key or default_dist_key
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
    def get_stats_dict(cls, **data_dicts):
        stats = {}
        for key, value in data_dicts.items():
            stats[key] = stats.get(key, {})
            labels = value["labels"]
            for (lab, cnt) in zip(*np.unique(labels, return_counts=True)):
                stats[key].update(
                    {
                        str(lab): {
                            "count": str(cnt),
                            "percent": str(
                                round((cnt / len(labels)) * 100, 2)
                            ),
                        }
                    }
                )
        return stats

    @classmethod
    @timeit("Generating corpus for dataset", "Corpus generated")
    def generate_corpus(cls, docs, path):
        corpus_file = join(path, "_corpus.csv")
        if exists(corpus_file):
            corpus = corpus_from_csv(path=corpus_file)
        else:
            corpus = corpus_from_docs(docs)
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
        self.write_stats_json(
            self.path, train=self.__train_dict, test=self.__test_dict
        )
        self.__all_docs = set(
            self.__train_dict["sentences"] + self.__test_dict["sentences"]
        )
        self.__corpus = self.generate_corpus(self.__all_docs, self.path)

    def _set_dist_key(self, data):
        labels = [label for label in data["labels"]]
        _, counts = np.unique(labels, return_counts=True)
        total = np.sum(counts)
        dist_values = np.round(np.divide(counts, total) * 100).astype(np.int32)
        dist_key = "_".join([str(int(v * 100)) for v in dist_values])
        return dist_key

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
    def write_stats_json(cls, path, **data_dicts):
        stats = cls.get_stats_dict(**data_dicts)
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

                    if key == "train":
                        self.__train_dict = resampled_dict
                        self.__train_dist_key = dist_folder
                    else:
                        self.__test_dict = resampled_dict
                        self.__test_dist_key = dist_folder

                Dataset.write_stats_json(dist_path, **data_dicts)
                Dataset.generate_corpus(set(all_docs), dist_path)

        self.__all_docs = set(
            self.__train_dict["sentences"] + self.__test_dict["sentences"]
        )
        self.__corpus = self.generate_corpus(self.__all_docs, dist_path)
