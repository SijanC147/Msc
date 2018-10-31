from os import makedirs
from os.path import join, exists, basename, normpath
from operator import itemgetter
import numpy as np
import spacy
from spacy.attrs import ORTH  # pylint: disable=E0611
from tsaplay.features import FeatureProvider
from tsaplay.utils.data import (
    resample_data_dict,
    class_dist_info,
    class_dist_stats,
)
from tsaplay.utils.io import search_dir, unpickle_file, pickle_file, dump_json
from tsaplay.utils.decorators import timeit
from tsaplay.constants import DATASET_DATA_PATH, SPACY_MODEL


class Dataset:
    def __init__(self, name, distribution=None, data_root=None):
        self._data_root = data_root or DATASET_DATA_PATH
        self._gen_dir = join(self._data_root, name)
        if not exists(self._gen_dir):
            raise ValueError(
                """Expected name to be one of {0}, got {1}.
                Import new datasets using
                tsaplay.scripts.import_dataset""".format(
                    self.list_installed_datasets(self._data_root), name
                )
            )
        self._name = name
        train_dict_path = join(self.gen_dir, "_train_dict.pkl")
        test_dict_path = join(self.gen_dir, "_test_dict.pkl")
        stats_file_path = join(self.gen_dir, "_stats.json")
        train_corpus_path = join(self.gen_dir, "_train_corpus.pkl")
        test_corpus_path = join(self.gen_dir, "_test_corpus.pkl")
        self._train_dict = unpickle_file(path=train_dict_path)
        self._test_dict = unpickle_file(path=test_dict_path)
        if distribution:
            self._redistribute_data(distribution)
        if exists(train_corpus_path):
            self._train_corpus = unpickle_file(path=train_corpus_path)
        else:
            train_docs = self._train_dict["sentences"]
            self._train_corpus = self.generate_corpus(set(train_docs))
            pickle_file(path=train_corpus_path, data=self._train_corpus)
        if exists(test_corpus_path):
            self._test_corpus = unpickle_file(path=test_corpus_path)
        else:
            test_docs = self._test_dict["sentences"]
            self._test_corpus = self.generate_corpus(set(test_docs))
            pickle_file(path=test_corpus_path, data=self._test_corpus)

        labels = self._train_dict["labels"] + self._test_dict["labels"]
        self._class_labels = list(set(labels))

        dump_json(
            path=stats_file_path,
            data=class_dist_stats(
                classes=self._class_labels,
                train=self._train_dict,
                test=self._test_dict,
            ),
        )
        train_dist_data = class_dist_info(self._train_dict["labels"])
        test_dist_data = class_dist_info(self._test_dict["labels"])
        train_dist_key = "_".join(map(str, train_dist_data[2]))
        test_dist_key = "_".join(map(str, test_dist_data[2]))
        self._dist_key = (
            train_dist_key
            if train_dist_key == test_dist_key
            else "-".join([train_dist_key, test_dist_key])
        )
        self._uid = "-".join([self._name, self._dist_key])

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
    def train_dict(self):
        return self._train_dict

    @property
    def test_dict(self):
        return self._test_dict

    @property
    def train_corpus(self):
        return self._train_corpus

    @property
    def test_corpus(self):
        return self._test_corpus

    @property
    def class_labels(self):
        return self._class_labels

    @property
    def dist_key(self):
        return self._dist_key

    @classmethod
    def list_installed_datasets(cls, data_root=DATASET_DATA_PATH):
        return [
            basename(normpath(path))
            for path in search_dir(data_root, kind="folders")
        ]

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
        words.sort(key=itemgetter(1), reverse=True)
        for word_id, cnt in words:
            corpus[nlp.vocab.strings[word_id]] = cnt
        return corpus

    @timeit("Redistributing dataset", "Dataset redistributed")
    def _redistribute_data(self, distribution):
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
            dist_dict_path = join(dist_path, "_" + key + "_dict.pkl")
            dist_corpus_path = join(dist_path, "_" + key + "_corpus.pkl")
            if exists(dist_dict_path):
                resampled_dict = unpickle_file(path=dist_dict_path)
                if key == "train":
                    self._train_dict = resampled_dict
                else:
                    self._test_dict = resampled_dict
            else:
                data_dicts = {}
                for mode in ["train", "test"]:
                    dist_dict_path = join(dist_path, "_" + mode + "_dict.pkl")
                    if mode == "train":
                        orig_dict = self._train_dict
                    else:
                        orig_dict = self._test_dict

                    resampled_dict = resample_data_dict(orig_dict, dist_values)
                    resampled_docs = set(resampled_dict["sentences"])
                    resampled_corpus = self.generate_corpus(resampled_docs)
                    pickle_file(path=dist_dict_path, data=resampled_dict)
                    pickle_file(path=dist_corpus_path, data=resampled_corpus)
                    data_dicts[mode] = resampled_dict

                dump_json(
                    path=join(dist_path, "_stats.json"),
                    data=class_dist_stats(
                        classes=self.class_labels, **data_dicts
                    ),
                )

                if key == "train":
                    self._train_dict = data_dicts[key]
                else:
                    self._test_dict = data_dicts[key]
