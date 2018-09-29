from itertools import chain
from collections import defaultdict
from os.path import join, exists

from tsaplay.utils._nlp import corpus_from_docs
from tsaplay.utils._io import (
    corpus_from_csv,
    corpus_to_csv,
    unpickle_file as _unpickle,
    pickle_file as _pickle,
)
import tsaplay.datasets._constants as DATASETS


class CompoundDataset:
    def __init__(self, datasets):
        self.__dsts = datasets

    @property
    def name(self):
        return "_".join(sorted([ds.name.lower() for ds in self.__dsts]))

    @property
    def path(self):
        return join(DATASETS.DATA_DIR, self.name)

    @property
    def datasets(self):
        return self.__dsts

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

    def _initialize_all_internals(self):
        self.__train_dict = self._load_dataset_dictionary(dict_type="train")
        self.__test_dict = self._load_dataset_dictionary(dict_type="test")
        self.__all_docs = set(
            self.__train_dict["sentences"] + self.__test_dict["sentences"]
        )
        self.__corpus = self._generate_corpus()

    def _load_dataset_dictionary(self, dict_type):
        if dict_type == "train":
            dict_file = join(self.path, "_train_dict.pkl")
        else:
            dict_file = join(self.path, "_test_dict.pkl")
        if exists(dict_file):
            dictionary = _unpickle(path=dict_file)
        else:
            if dict_type == "train":
                dict_list = [ds.train_dict for ds in self.__dsts]
            else:
                dict_list = [ds.test_dict for ds in self.__dsts]
            dictionary = CompoundDataset.concat_dicts(dict_list)
            _pickle(path=dict_file, data=dictionary)
        return dictionary

    def _generate_corpus(self):
        corpus_file = join(self.path, "_corpus.csv")
        if exists(corpus_file):
            corpus = corpus_from_csv(path=corpus_file)
        else:
            corpus = corpus_from_docs(docs=self.all_docs)
            corpus_to_csv(corpus_file, corpus)
        return corpus

    @classmethod
    def concat_dicts(cls, dicts_list):
        new_dict = defaultdict(list)
        for k, v in chain.from_iterable([d.items() for d in dicts_list]):
            new_dict[k] = new_dict[k] + v
        return dict(new_dict)
