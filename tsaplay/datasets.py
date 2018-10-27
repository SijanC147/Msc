from os import makedirs
from os.path import join, exists, basename, normpath
import json
import numpy as np
import spacy
from spacy.attrs import ORTH  # pylint: disable=E0611
from tsaplay.features import FeatureProvider
from tsaplay.utils.data import resample_data_dict, get_class_distribution
from tsaplay.utils.io import (
    search_dir,
    corpus_from_csv,
    corpus_to_csv,
    unpickle_file as _unpickle,
    pickle_file as _pickle,
)
from tsaplay.utils.decorators import timeit
from tsaplay.constants import DATASET_DATA_PATH, SPACY_MODEL


class Dataset:
    def __init__(self, name, distribution=None, data_root=None):
        self._data_root = data_root or DATASET_DATA_PATH
        self._gen_dir = join(self._data_root, name)
        if exists(self._gen_dir):
            self.name = name
            self._initialize_all_internals()
        else:
            raise ValueError(
                """Expected name to be on of {0}, got {1}.
                Import new datasets using 
                tsaplay.scripts.import_dataset""".format(
                    self.list_installed_datasets(self._data_root), name
                )
            )
        if distribution is not None:
            self._redistribute_data(distribution)
        else:
            train_dist_data = get_class_distribution(self.train_dict["labels"])
            test_dist_data = get_class_distribution(self.train_dict["labels"])
            self.__train_dist_key = "_".join(map(str, train_dist_data[2]))
            self.__test_dist_key = "_".join(map(str, test_dist_data[2]))

    @property
    def gen_dir(self):
        return self._gen_dir

    @property
    def train_dist_key(self):
        return self.__train_dist_key

    @property
    def test_dist_key(self):
        return self.__test_dist_key

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
    def list_installed_datasets(cls, data_root=DATASET_DATA_PATH):
        return [
            basename(normpath(path))
            for path in search_dir(data_root, kind="folders")
        ]

    @classmethod
    @timeit("Generating dataset stats", "Dataset stats generated")
    def get_stats_dict(cls, classes=None, **data_dicts):
        stats = {}
        for key, value in data_dicts.items():
            stats[key] = stats.get(key, {})
            dist_data = get_class_distribution(
                value["labels"], all_classes=classes
            )
            for (_class, count, dist) in zip(*dist_data):
                stats[key].update(
                    {str(_class): {"count": str(count), "percent": str(dist)}}
                )
        return stats

    @classmethod
    def write_corpus_file(cls, docs, path):
        corpus_file = join(path, "_corpus.pkl")
        if exists(corpus_file):
            return _unpickle(corpus_file)
        corpus = cls.generate_corpus(docs)
        _pickle(data=corpus, path=corpus_file)
        return corpus

    @classmethod
    @timeit("Generating corpus for dataset", "Corpus generated")
    def generate_corpus(cls, docs):
        corpus = {}

        nlp = spacy.load(SPACY_MODEL, disable=["parser", "ner"])
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

    def _initialize_all_internals(self):
        self.__train_dict = _unpickle(path=join(self.gen_dir, "_train.pkl"))
        self.__test_dict = _unpickle(path=join(self.gen_dir, "_test.pkl"))
        self.__default_classes = np.unique(
            [label for label in self.__train_dict["labels"]]
        ).tolist()
        self.write_stats_json(
            path=self.gen_dir,
            classes=self.__default_classes,
            train=self.__train_dict,
            test=self.__test_dict,
        )
        self.__all_docs = set(
            self.__train_dict["sentences"] + self.__test_dict["sentences"]
        )
        self.__corpus = self.write_corpus_file(self.__all_docs, self.gen_dir)

    @classmethod
    def write_stats_json(cls, path, classes=None, **data_dicts):
        stats_file_path = join(path, "_stats.json")
        if not exists(stats_file_path):
            stats = cls.get_stats_dict(classes=classes, **data_dicts)
            with open(stats_file_path, "w+") as stats_file:
                json.dump(stats, stats_file, indent=4)

    @timeit("Redistributing dataset", "Dataset redistributed")
    def _redistribute_data(self, distribution):
        self.__train_dist_key = None
        self.__test_dist_key = None
        dists_dir = join(self.gen_dir, "_dists")
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
                self.write_corpus_file(set(all_docs), dist_path)

                if key == "train":
                    self.__train_dict = data_dicts[key]
                    self.__train_dist_key = dist_folder
                elif key == "test":
                    self.__test_dict = data_dicts[key]
                    self.__test_dist_key = dist_folder

        self.__all_docs = set(
            self.__train_dict["sentences"] + self.__test_dict["sentences"]
        )
        self.__corpus = self.write_corpus_file(self.__all_docs, dist_path)
