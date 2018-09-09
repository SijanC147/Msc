import os
import csv
import spacy
import time
import math
import random
from os import listdir, makedirs
from os.path import isfile, join, exists
from statistics import mean
from utils import (
    token_filter,
    re_dist,
    inspect_dist,
    search_dir,
    corpus_from_docs,
    corpus_from_csv,
    corpus_to_csv,
    write_embedding_to_disk,
    write_emb_tsv_to_disk,
    get_sentence_contexts,
    unpickle_file as unpickle,
    pickle_file as pickle,
)
from spacy.tokens import Doc
from spacy.attrs import ORTH  # pylint: disable=E0611
from abc import ABC, abstractmethod


class Dataset(ABC):
    def __init__(self, path, embedding=None):
        self.__path = None
        self.path = path
        self.embedding = embedding
        # if len(parent_folder) == 0:
        #     parent_folder = self.__class__.__name__
        # else:
        #     parent_folder = os.path.join(
        #         self.__class__.__name__, parent_folder
        #     )
        # self.parent_directory = os.path.join(
        #     os.path.dirname(os.path.abspath(__file__)), "data", parent_folder
        # )
        # self.train_file_path = os.path.join(
        #     self.parent_directory, train_file_path
        # )
        # self.eval_file_path = os.path.join(
        #     self.parent_directory, eval_file_path
        # )
        # self.debug_file_path = os.path.join(
        #     self.parent_directory, debug_file_path
        # )
        # self.generated_data_directory = os.path.join(
        #     self.parent_directory, "generated"
        # )
        # self.corpus_file_path = os.path.join(
        #     self.parent_directory, self.generated_data_directory, "corpus.csv"
        # )
        # self.vocabulary_corpus = self.get_vocabulary_corpus(
        #     rebuild=rebuild_corpus
        # )
        # if embedding is not None:
        #     self.set_embedding(embedding)

    @abstractmethod
    def _parse_file(self, file):
        pass

    @abstractmethod
    def generate_dataset_dictionary(self, mode):
        pass

    @property
    def embedding(self):
        return self.__embedding

    @property
    def path(self):
        return self.__path

    @property
    def train_file(self):
        return search_dir(
            dir=self.path, query="train", first=True, files_only=True
        )

    @property
    def test_file(self):
        return search_dir(
            dir=self.path, query="test", first=True, files_only=True
        )

    @property
    def gen_dir(self):
        gen_dir = join(self.path, "generated")
        makedirs(gen_dir, exist_ok=True)
        return gen_dir

    @property
    def corpus(self):
        corpus_file = join(self.gen_dir, "corpus.csv")
        if exists(corpus_file):
            self.__corpus = corpus_from_csv(path=corpus_file)
        else:
            self.corpus = corpus_from_docs(docs=self.all_docs)

    @property
    def all_docs(self):
        if self.__all_docs is None:
            train_docs = self.parse_file(self.train_file)["sentences"]
            test_docs = self.parse_file(self.test_file)["sentences"]
            self.__all_docs = set(train_docs + test_docs)

        return self.__all_docs

    @property
    def train_dict(self):
        if self.__train_dict is None:
            train_dict_file = join(self.gen_dir, "train_dict.pkl")
            if exists(train_dict_file):
                self.__train_dict = unpickle(path=train_dict_file)
            else:
                self.train_dict = self._parse_file(self.train_file)
        return self.__train_dict

    @property
    def test_dict(self):
        if self.__test_dict is None:
            test_dict_file = join(self.gen_dir, "test_dict.pkl")
            if exists(test_dict_file):
                self.__test_dict = unpickle(path=test_dict_file)
            else:
                self.test_dict = self._parse_file(self.test_file)
        return self.__test_dict

    @property
    def train_features_and_labels(self):
        if self.__train_features_and_labels is None:
            save_file = join(self.__emb_gen_dir, "train.pkl")
            if exists(save_file):
                self.__train_features_and_labels = unpickle(path=save_file)
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
                self.__test_features_and_labels = unpickle(path=save_file)
            else:
                self.test_features_and_labels = self._parse_features_labels(
                    dictionary=self.test_dict
                )
        return self.__test_features_and_labels

    @path.setter
    def path(self, path):
        if self.__path != path:
            self.__path = path
            self.__all_docs = None
            self.__corpus = None

    @embedding.setter
    def embedding(self, embedding):
        if embedding is not None:
            name = type(embedding).__name__
            version = embedding.alias
            emb_gen_dir = join(self.gen_dir, name, version)
            makedirs(emb_gen_dir, exist_ok=True)
            partial_name = "partial_{v}.txt".format({"v": version})
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
            pickle(path=train_dict_file, data=train_dict)
        self.__train_dict = train_dict

    @test_dict.setter
    def test_dict(self, test_dict):
        if test_dict is not None:
            test_dict_file = join(self.gen_dir, "test_dict.pkl")
            pickle(path=test_dict_file, data=test_dict)
        self.__test_dict = test_dict

    @train_features_and_labels.setter
    def train_features_and_labels(self, values):
        if values is not None:
            save_file = join(self.__emb_gen_dir, "train.pkl")
            pickle(path=save_file, data=values)
        self.__train_features_and_labels = values

    @test_features_and_labels.setter
    def test_features_and_labels(self, values):
        if values is not None:
            save_file = join(self.__emb_gen_dir, "test.pkl")
            pickle(path=save_file, data=values)
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

    def _parse_features_labels(self, dictionary, distribution=None):
        features = {
            "sentence": [],
            "sentence_length": [],
            "target": [],
            "mappings": {"left": [], "target": [], "right": []},
        }
        labels = []

        total_time = 0
        for index in range(len(dictionary["sentences"])):

            start = time.time()

            sentence = dictionary["sentences"][index].strip()
            target = dictionary["targets"][index].strip()
            label = (
                int(dictionary["labels"][index].strip())
                if type(dictionary["labels"][index]) == str
                else dictionary["labels"][index]
            )

            features["sentence"].append(sentence)
            features["target"].append(target)
            labels.append(label)
            left_context, right_context = get_sentence_contexts(
                sentence=sentence,
                target=target,
                offset=dictionary.get("offset"),
            )

            left_mapping = self.embedding.get_index_ids(left_context)
            target_mapping = self.embedding.get_index_ids(target)
            right_mapping = self.embedding.get_index_ids(right_context)

            features["sentence_length"].append(
                len(left_mapping + target_mapping + right_mapping)
            )

            features["mappings"]["left"].append(left_mapping)
            features["mappings"]["target"].append(target_mapping)
            features["mappings"]["right"].append(right_mapping)

            total_time += time.time() - start
            if index % 60 == 0:
                print(
                    "Processed {0}/{1} lines ({2:.2f}%) tot:{3:.3f}s avg:{4:.3f}s/line".format(
                        index + 1,
                        len(dictionary["sentences"]),
                        ((index + 1) / len(dictionary["sentences"])) * 100,
                        total_time,
                        total_time / (index + 1),
                    )
                )

        return {"features": features, "labels": labels}

    # def get_all_text_in_dataset(self):
    #     return set(
    #         self.get_dataset_dictionary(mode="train")["sentences"]
    #         + self.get_dataset_dictionary(mode="eval")["sentences"]
    #     )

    # def get_left_and_right_contexts(self, sentence, target, offset=None):
    #     if offset is None:
    #         left, _, right = sentence.partition(target)
    #     else:
    #         left = sentence[:offset]
    #         start = offset + len(target)
    #         right = sentence[start:]
    #     return left.strip(), right.strip()

    # def get_file(self, mode):
    #     if mode == "train":
    #         return self.train_file_path
    #     elif mode == "eval":
    #         return self.eval_file_path
    #     else:
    #         return self.debug_file_path

    # def set_embedding(self, embedding):
    #     self.embedding = embedding
    #     self.generated_embedding_directory = os.path.join(
    #         self.generated_data_directory,
    #         type(self.embedding).__name__,
    #         self.embedding.alias,
    #     )
    #     self.embedding_id_file_path = os.path.join(
    #         self.generated_embedding_directory, "words_to_ids.csv"
    #     )
    #     self.projection_labels_file_path = os.path.join(
    #         self.generated_embedding_directory,
    #         "tensorboard_projection_labels.tsv",
    #     )
    #     self.partial_embedding_file_path = os.path.join(
    #         self.generated_embedding_directory,
    #         "partial_" + self.embedding.version + ".txt",
    #     )
    #     if self.partial_embedding_file_exists():
    #         self.embedding.path = self.partial_embedding_file_path

    # def get_save_file_path(self, mode):
    #     return os.path.join(
    #         self.generated_embedding_directory,
    #         "features_labels_" + mode + ".pkl",
    #     )

    # def get_dataset_dictionary_file_path(self, mode):
    #     return os.path.join(
    #         self.generated_data_directory,
    #         "raw_dataset_dictionary_" + mode + ".pkl",
    #     )

    # def corpus_file_exists(self):
    #     return os.path.isfile(self.corpus_file_path)

    # def embedding_id_file_exists(self):
    #     return os.path.exists(self.embedding_id_file_path)

    # def partial_embedding_file_exists(self):
    #     return os.path.exists(self.partial_embedding_file_path)

    # def projection_labels_file_exists(self):
    #     return os.path.exists(self.projection_labels_file_path)

    # def features_labels_save_file_exists(self, mode):
    #     return os.path.exists(self.get_save_file_path(mode))

    # def dataset_dictionary_file_exists(self, mode):
    #     return os.path.exists(self.get_dataset_dictionary_file_path(mode))

    # def load_features_and_labels_from_save(self, mode):
    #     if self.features_labels_save_file_exists(mode):
    #         with open(self.get_save_file_path(mode), "rb") as f:
    #             saved_data = pickle.load(f)
    #             features = saved_data["features"]
    #             labels = saved_data["labels"]
    #             return features, labels

    # def save_features_and_labels(self, mode, features, labels):
    #     saved_data = {"features": features, "labels": labels}

    #     with open(self.get_save_file_path(mode), "wb+") as f:
    #         pickle.dump(saved_data, f, pickle.HIGHEST_PROTOCOL)

    # def save_dataset_dictionary(self, dataset_dictionary, mode):
    #     os.makedirs(self.generated_data_directory, exist_ok=True)
    #     with open(self.get_dataset_dictionary_file_path(mode), "wb+") as f:
    #         pickle.dump(dataset_dictionary, f, pickle.HIGHEST_PROTOCOL)

    # def load_vocabulary_corpus_from_csv(self):
    #     vocabulary_corpus = {}
    #     if self.corpus_file_exists():
    #         with open(self.corpus_file_path) as csvfile:
    #             reader = csv.DictReader(csvfile)
    #             for row in reader:
    #                 vocabulary_corpus[row["word"]] = int(row["count"])
    #     return vocabulary_corpus

    # def load_dataset_dictionary_from_save(self, mode):
    #     if self.dataset_dictionary_file_exists(mode):
    #         with open(self.get_dataset_dictionary_file_path(mode), "rb") as f:
    #             return pickle.load(f)

    # def generate_vocabulary_corpus(self, source_documents):
    #     vocabulary_corpus = {}

    #     nlp = spacy.load("en")
    #     tokens = nlp(
    #         " ".join(map(lambda document: document.strip(), source_documents))
    #     )
    #     filtered_tokens = list(filter(token_filter, tokens))
    #     filtered_doc = nlp(
    #         " ".join(map(lambda token: token.text, filtered_tokens))
    #     )
    #     counts = filtered_doc.count_by(ORTH)
    #     os.makedirs(self.generated_data_directory, exist_ok=True)
    #     with open(self.corpus_file_path, "w") as csvfile:
    #         writer = csv.DictWriter(csvfile, fieldnames=["word", "count"])

    #         writer.writeheader()
    #         for word_id, count in sorted(
    #             counts.items(), reverse=True, key=lambda item: item[1]
    #         ):
    #             writer.writerow(
    #                 {"word": nlp.vocab.strings[word_id], "count": count}
    #             )
    #             vocabulary_corpus[nlp.vocab.strings[word_id]] = count

    #     return vocabulary_corpus

    # def generate_partial_embedding_file(self, partial_embedding):
    #     os.makedirs(self.generated_embedding_directory, exist_ok=True)
    #     with open(self.partial_embedding_file_path, "w+") as f:
    #         for word in [*partial_embedding]:
    #             if word != "<OOV>" and word != "<PAD>":
    #                 f.write(
    #                     word
    #                     + " "
    #                     + " ".join(partial_embedding[word].astype(str))
    #                     + "\n"
    #                 )

    # def generate_projection_labels_file(self, partial_embedding):
    #     os.makedirs(self.generated_embedding_directory, exist_ok=True)
    #     with open(self.projection_labels_file_path, "w+") as f:
    #         f.write("Words\n")
    #         for word in [*partial_embedding]:
    #             f.write(word + "\n")

    # def get_vocabulary_corpus(self, rebuild=False):
    #     if self.corpus_file_exists() and not (rebuild):
    #         return self.load_vocabulary_corpus_from_csv()
    #     else:
    #         return self.generate_vocabulary_corpus(
    #             self.get_all_text_in_dataset()
    #         )

    # def get_dataset_dictionary(self, mode):
    #     if self.dataset_dictionary_file_exists(mode):
    #         return self.load_dataset_dictionary_from_save(mode)
    #     else:
    #         dataset_dictionary = self.generate_dataset_dictionary(mode)
    #         self.save_dataset_dictionary(
    #             dataset_dictionary=dataset_dictionary, mode=mode
    #         )
    #         return dataset_dictionary

    # def load_embedding_from_corpus(
    #     self, corpus, rebuild_partial=False, rebuild_projection=False
    # ):
    #     if self.partial_embedding_file_exists() and not rebuild_partial:
    #         self.embedding.path = self.partial_embedding_file_path
    #     else:
    #         self.embedding.filter_on_vocab(vocab=[*corpus])
    #         self.generate_partial_embedding_file(self.embedding.dictionary)
    #     if not (self.projection_labels_file_exists()) or rebuild_projection:
    #         self.generate_projection_labels_file(self.embedding.dictionary)
