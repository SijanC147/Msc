from time import time as _time
from os import listdir, makedirs
from os.path import normpath, basename, isfile, join, exists, dirname
from statistics import mean
from functools import wraps
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
    def __init__(self, path, parser, embedding=None, filter_embedding=True):
        self.path = path
        self.parser = parser
        self.filter_embedding = filter_embedding
        self.embedding = embedding

    @property
    def path(self):
        return self.__path

    @property
    def parser(self):
        return self.__parser

    @property
    def embedding(self):
        return self.__embedding

    @property
    def name(self):
        return basename(normpath(self.path))

    @property
    def train_file(self):
        file_name = search_dir(
            dir=self.path, query="train", first=True, files_only=True
        )
        return join(self.path, file_name)

    @property
    def test_file(self):
        file_name = search_dir(
            dir=self.path, query="test", first=True, files_only=True
        )
        return join(self.path, file_name)

    @property
    def gen_dir(self):
        gen_dir = join(self.path, "_generated")
        makedirs(gen_dir, exist_ok=True)
        return gen_dir

    @property
    def corpus(self):
        if self.__corpus is None:
            corpus_file = join(self.gen_dir, "corpus.csv")
            if exists(corpus_file):
                self.__corpus = corpus_from_csv(path=corpus_file)
            else:
                self.corpus = corpus_from_docs(docs=self.all_docs)

        return self.__corpus

    @property
    def all_docs(self):
        if self.__all_docs is None:
            train_docs = self._parse_file(self.train_file)["sentences"]
            test_docs = self._parse_file(self.test_file)["sentences"]
            self.__all_docs = set(train_docs + test_docs)

        return self.__all_docs

    @property
    def train_dict(self):
        if self.__train_dict is None:
            train_dict_file = join(self.gen_dir, "train_dict.pkl")
            if exists(train_dict_file):
                self.__train_dict = _unpickle(path=train_dict_file)
            else:
                self.train_dict = self._parse_file(self.train_file)
        return self.__train_dict

    @property
    def test_dict(self):
        if self.__test_dict is None:
            test_dict_file = join(self.gen_dir, "test_dict.pkl")
            if exists(test_dict_file):
                self.__test_dict = _unpickle(path=test_dict_file)
            else:
                self.test_dict = self._parse_file(self.test_file)
        return self.__test_dict

    @property
    def train_features_and_labels(self):
        if self.__train_features_and_labels is None:
            save_file = join(self.__emb_gen_dir, "train.pkl")
            if exists(save_file):
                self.__train_features_and_labels = _unpickle(path=save_file)
            else:
                self.train_features_and_labels = self._parse_features_labels(
                    dictionary=self.train_dict
                )
        return self.__train_features_and_labels

    @property
    def test_features_and_labels(self):
        if self.__test_features_and_labels is None:
            save_file = join(self.__emb_gen_dir, "test.pkl")
            if exists(save_file):
                self.__test_features_and_labels = _unpickle(path=save_file)
            else:
                self.test_features_and_labels = self._parse_features_labels(
                    dictionary=self.test_dict
                )
        return self.__test_features_and_labels

    @path.setter
    def path(self, path):
        try:
            path_changed = self.__path != path
        except:
            path_changed = True

        if path_changed:
            self._reset(path)

    @parser.setter
    def parser(self, parser):
        self.__parser = self._wrap_parser(parser)

    @embedding.setter
    def embedding(self, embedding):
        if embedding is not None:
            if self.filter_embedding:
                name = embedding.name
                version = embedding.version
                emb_gen_dir = join(self.gen_dir, name)
                makedirs(emb_gen_dir, exist_ok=True)
                partial_name = "partial_{0}.txt".format(version)
                partial_path = join(emb_gen_dir, partial_name)
                tb_tsv_path = join(emb_gen_dir, "projection_meta.tsv")
                if exists(partial_path):
                    embedding.path = partial_path
                else:
                    embedding.filter_on_vocab(self.corpus)
                    write_embedding_to_disk(
                        path=partial_path, emb_dict=embedding.dictionary
                    )
                if not exists(tb_tsv_path):
                    write_emb_tsv_to_disk(
                        path=tb_tsv_path, emb_dict=embedding.dictionary
                    )
            else:
                name = embedding.name
                version = embedding.version
                emb_gen_dir = join(self.gen_dir, name)
                makedirs(emb_gen_dir, exist_ok=True)
                # emb_gen_dir = join(
                #     dirname(embedding.path), "_generated", self.name
                # )
            self.__emb_gen_dir = emb_gen_dir
        self.__embedding = embedding

    @corpus.setter
    def corpus(self, corpus):
        if corpus is not None:
            corpus_file = join(self.gen_dir, "corpus.csv")
            corpus_to_csv(path=corpus_file, corpus=corpus)

        self.__corpus = corpus

    @train_dict.setter
    def train_dict(self, train_dict):
        if train_dict is not None:
            train_dict_file = join(self.gen_dir, "train_dict.pkl")
            _pickle(path=train_dict_file, data=train_dict)
        self.__train_dict = train_dict

    @test_dict.setter
    def test_dict(self, test_dict):
        if test_dict is not None:
            test_dict_file = join(self.gen_dir, "test_dict.pkl")
            _pickle(path=test_dict_file, data=test_dict)
        self.__test_dict = test_dict

    @train_features_and_labels.setter
    def train_features_and_labels(self, values):
        if values is not None:
            save_file = join(self.__emb_gen_dir, "train.pkl")
            _pickle(path=save_file, data=values)
        self.__train_features_and_labels = values

    @test_features_and_labels.setter
    def test_features_and_labels(self, values):
        if values is not None:
            save_file = join(self.__emb_gen_dir, "test.pkl")
            _pickle(path=save_file, data=values)
        self.__test_features_and_labels = values

    def get_features_and_labels(self, mode, distribution=None):
        if mode == "train":
            data = self.train_features_and_labels
        elif mode == "eval" or mode == "test":
            data = self.test_features_and_labels

        features = data["features"]
        labels = data["labels"]

        if distribution is not None:
            features, labels = re_dist(
                features=features, labels=labels, distribution=distribution
            )
        stats = inspect_dist(features=features, labels=labels)

        return features, labels, stats

    def _reset(self, path):
        self.__path = path
        self.__all_docs = None
        self.__corpus = None
        self.__train_dict = None
        self.__test_dict = None
        self.__train_features_and_labels = None
        self.__test_features_and_labels = None

    def _parse_file(self, path):
        return self.parser(path)

    def _parse_features_labels(self, dictionary, distribution=None):
        features = {
            "sentence": [],
            "sentence_length": [],
            "left": [],
            "target": [],
            "right": [],
            "mappings": {"left": [], "target": [], "right": []},
        }
        labels = []

        print("Processing dataset...")
        total_time = 0
        for index in range(len(dictionary["sentences"])):

            start = _time()

            single_feature = get_sentence_target_features(
                embedding=self.embedding,
                sentence=dictionary["sentences"][index],
                target=dictionary["targets"][index],
                label=dictionary["labels"][index],
                offset=dictionary.get("offset"),
            )
            features["sentence"].append(single_feature["sentence"])
            features["target"].append(single_feature["target_lit"])
            features["sentence_length"].append(single_feature["sentence_len"])
            features["left"].append(single_feature["left_lit"])
            features["right"].append(single_feature["right_lit"])
            features["mappings"]["left"].append(single_feature["left_map"])
            features["mappings"]["target"].append(single_feature["target_map"])
            features["mappings"]["right"].append(single_feature["right_map"])
            labels.append(single_feature["label"])

            total_time += _time() - start
            if index % 60 == 0:
                print(
                    "{0}/{1} ({2:.2f}%) tot:{3:.3f}s avg:{4:.3f}s/line".format(
                        index + 1,
                        len(dictionary["sentences"]),
                        ((index + 1) / len(dictionary["sentences"])) * 100,
                        total_time,
                        total_time / (index + 1),
                    )
                )

        return {"features": features, "labels": labels}

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
