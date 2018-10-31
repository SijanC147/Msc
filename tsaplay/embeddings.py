from os import makedirs
from os.path import join, exists, dirname
from math import sqrt, floor
from collections import namedtuple
from inspect import getsource, Parameter, signature, getmembers
from hashlib import md5
from warnings import warn
from functools import partial
from contextlib import redirect_stdout
from io import StringIO
from csv import writer
import tensorflow as tf
import numpy as np
import gensim.downloader as gensim_data
from gensim.models import KeyedVectors
import spacy
from spacy.language import Language
from spacy.tokens import Doc
from tsaplay.constants import EMBEDDING_DATA_PATH, SPACY_MODEL, RANDOM_SEED
from tsaplay.utils.decorators import timeit
from tsaplay.utils.io import cprnt, pickle_file, unpickle_file

FASTTEXT_WIKI_300 = "fasttext-wiki-news-subwords-300"
GLOVE_TWITTER_25 = "glove-twitter-25"
GLOVE_TWITTER_50 = "glove-twitter-50"
GLOVE_TWITTER_100 = "glove-twitter-100"
GLOVE_TWITTER_200 = "glove-twitter-200"
GLOVE_WIKI_GIGA_50 = "glove-wiki-gigaword-50"
GLOVE_WIKI_GIGA_100 = "glove-wiki-gigaword-100"
GLOVE_WIKI_GIGA_200 = "glove-wiki-gigaword-200"
GLOVE_WIKI_GIGA_300 = "glove-wiki-gigaword-300"
GLOVE_COMMON42_300 = "glove-cc42-300"
GLOVE_COMMON840_300 = "glove-cc840-300"
W2V_GOOGLE_300 = "word2vec-google-news-300"
W2V_RUS_300 = "word2vec-ruscorpora-300"

FilterDetails = namedtuple("FilterDetails", "hash filter reduction report")


class Embedding:
    def __init__(
        self, source, oov=None, max_shards=10, filters=None, data_root=None
    ):
        self._data_root = data_root or EMBEDDING_DATA_PATH
        self._oov = oov or self.default_oov
        if not callable(self._oov):
            warn("OOV parameter is not callable, falling back to default.")
            self._oov = self.default_oov
        self._source = source
        if filters:
            filters_list_str = list(map(self.filters_as_str, filters))
            filters_list_str = [self._source] + filters_list_str
            filters_list_md5 = [
                md5(filt.encode("utf-8")).hexdigest()
                for filt in filters_list_str
            ]
            filters_list_md5 = list(set(filters_list_md5))
            filters_list_md5.sort()
            self._filter_hash_id = md5(
                str(filters_list_md5).encode("utf-8")
            ).hexdigest()
            self._name = "--".join([source, self._filter_hash_id])
        else:
            self._filter_hash_id = None
            self._name = source
        self._gen_dir = join(self._data_root, self.name)
        makedirs(self._gen_dir, exist_ok=True)
        self._gensim_model, filter_details = self.load_gensim_model(
            source, filters, self._gen_dir
        )
        self._filter_details = (
            FilterDetails(
                self._filter_hash_id,
                filter_details.filter,
                filter_details.reduction,
                filter_details.report,
            )
            if filter_details
            else None
        )
        self._vectors = np.concatenate(
            [
                [np.zeros(shape=self.dim_size)],
                self._gensim_model.vectors,
            ]
        ).astype(np.float32)
        self._num_shards = self.ideal_partition_divisor(
            self.vocab_size, max_shards
        )
        self._vocab_file_path = self.export_vocab_files(
            self.vocab, self.gen_dir
        )

    @property
    def source(self):
        return self._source

    @property
    def name(self):
        return self._name

    @property
    def oov(self):
        return self._oov

    @property
    def gen_dir(self):
        return self._gen_dir

    @property
    def vocab(self):
        return ["<PAD>"] + self._gensim_model.index2word

    @property
    def dim_size(self):
        return self._gensim_model.vector_size

    @property
    def vocab_size(self):
        return len(self.vectors)

    @property
    def vocab_file_path(self):
        return self._vocab_file_path

    @property
    def filter_details(self):
        return self._filter_details

    @property
    def vectors(self):
        return self._vectors

    @property
    def num_shards(self):
        return self._num_shards

    def initializer_fn(self, structure=None):
        shape = (self.vocab_size, self.dim_size)
        partition_size = int(self.vocab_size / self.num_shards)

        def _init_var(shape=shape, dtype=tf.float32, partition_info=None):
            return self.vectors

        def _init_part_var(shape=shape, dtype=tf.float32, partition_info=None):
            part_offset = partition_info.single_offset(shape)
            this_slice = part_offset + partition_size
            return self.vectors[part_offset:this_slice]

        def _init_const():
            return self.vectors

        _init_fn = {
            "partitioned": _init_part_var,
            "constant": _init_const,
            "variable": _init_var,
        }.get(structure, _init_var)

        return _init_fn

    @classmethod
    def default_oov(cls, size):
        np.random.seed(RANDOM_SEED)
        return np.random.uniform(low=-0.03, high=0.03, size=size)

    @classmethod
    def filters_as_str(cls, filter_condition):
        if isinstance(filter_condition, str):
            return filter_condition
        if isinstance(filter_condition, partial):
            return str(filter_condition.keywords)
        if callable(filter_condition):
            return getsource(filter_condition)
        if hasattr(filter_condition, "sort"):
            filter_condition = list(set(map(str.lower, filter_condition)))
            filter_condition.sort()
        return str(filter_condition)

    @classmethod
    def ideal_partition_divisor(cls, vocab_size, max_shards):
        for i in range(max_shards, 0, -1):
            if vocab_size % i == 0:
                return i

    @classmethod
    def export_vocab_files(cls, vocab, export_dir, aux_tokens=None):
        vocab_file_path = join(export_dir, "_vocab.txt")
        tsv_file_path = join(export_dir, "_vocab.tsv")
        vocab = (aux_tokens or []) + vocab
        if not exists(vocab_file_path):
            with open(vocab_file_path, "w") as vocab_file:
                for word in vocab:
                    if word != "<PAD>":
                        vocab_file.write("{0}\n".format(word))
            with open(tsv_file_path, "w") as tsv_file:
                for word in vocab:
                    tsv_file.write("{0}\n".format(word))
        return vocab_file_path

    @classmethod
    def export_gensim_model(cls, gensim_model, export_dir, aux_tokens=None):
        model_bin_path = join(export_dir, "_gensim_model.bin")
        gensim_model.save(model_bin_path)
        default_aux_tokens = ["<PAD>", "<OOV>"]
        aux = default_aux_tokens + (aux_tokens or [])
        cls.export_vocab_files(gensim_model.index2word, export_dir, aux)

    @classmethod
    @timeit("Filtering embedding (this can take a while)", "Filtering done")
    def filter_gensim_model(cls, gensim_model, vocab_filters, detail_dir=None):
        entities = gensim_model.index2word
        filter_report = None

        function_filters = list(filter(callable, vocab_filters))
        if function_filters:
            nlp = spacy.load(SPACY_MODEL)
            default_pipes_param = Parameter(
                name="pipes",
                kind=Parameter.POSITIONAL_OR_KEYWORD,
                default=["dep", "ner", "pos"],
            )

            def make_doc(vocab):
                return Doc(nlp.vocab, words=vocab)

            embedding_lang = Language(nlp.vocab, make_doc=make_doc)

            required_pipes = [
                signature(filter_fn)
                .parameters.get("pipes", default_pipes_param)
                .default
                for filter_fn in function_filters
            ]
            required_pipes = list(set(sum(required_pipes, [])))

            if "dep" in required_pipes:
                dep = spacy.pipeline.DependencyParser(nlp.vocab)
                dep.from_disk(join(SPACY_MODEL, "parser"))
                embedding_lang.add_pipe(dep)
            if "ner" in required_pipes:
                ner = spacy.pipeline.EntityRecognizer(nlp.vocab)
                ner.from_disk(join(SPACY_MODEL, "ner"))
                embedding_lang.add_pipe(ner)
            if "pos" in required_pipes:
                tagger = spacy.pipeline.Tagger(nlp.vocab)
                tagger = tagger.from_disk(join(SPACY_MODEL, "tagger"))
                embedding_lang.add_pipe(tagger)

            def grouped(token):
                keep = True
                for filter_fn in function_filters:
                    keep = keep and filter_fn(token)
                    if not keep:
                        params = signature(filter_fn).parameters
                        pipes = params["pipes"].default
                        if pipes == [None]:
                            attrs = params["attrs"].default
                            get_attrs = [
                                attr.replace("!", "") for attr in attrs
                            ]
                            pipes = ["None"]
                        elif pipes == ["pos"]:
                            attrs = params["tags"].default
                            get_attrs = ["pos_"]
                        elif pipes == ["ner"]:
                            attrs = params["tags"].default
                            get_attrs = ["ent_type"]
                        elif pipes == ["dep"]:
                            attrs = params["tags"].default
                            get_attrs = ["dep_"]

                        pipes_param = " ".join(pipes)
                        attrs_param = " ".join(attrs)
                        token_attrs = getmembers(
                            token,
                            predicate=lambda m: isinstance(
                                m, (str, float, int, bool)
                            ),
                        )
                        token_attrs = " ".join(
                            [
                                "{0}:{1}".format(n, v)
                                for n, v in token_attrs
                                if n in get_attrs
                            ]
                        )
                        print(
                            "<SEP>".join(
                                [
                                    filter_fn.func.__name__,
                                    pipes_param,
                                    attrs_param,
                                    token.text,
                                    token_attrs,
                                ]
                            )
                        )
                        return keep
                return keep

            embedding_lang.max_length = len(entities) + 1
            doc = embedding_lang(entities)

            filter_report = StringIO()
            with redirect_stdout(filter_report):
                entities = [token.text for token in filter(grouped, doc)]

            filter_report = filter_report.getvalue().split("\n")
            filter_report = [row.split("<SEP>") for row in filter_report]

        list_filters = filter(lambda cond: not callable(cond), vocab_filters)
        list_filters = sum(list_filters, [])
        if list_filters:
            entities = list(set(entities) & set(list_filters))

        weights = [gensim_model.get_vector(entity) for entity in entities]
        filtered_model = KeyedVectors(gensim_model.vector_size)
        filtered_model.add(entities, weights)

        if detail_dir:
            filter_str, reduction_percent = cls.export_filtered_details(
                gensim_model,
                filtered_model,
                filter_report,
                vocab_filters,
                detail_dir,
            )

        cprnt("Embedding size reduced by {}%".format(reduction_percent))
        filter_details = FilterDetails(
            None, filter_str, reduction_percent, filter_report
        )
        return filtered_model, filter_details

    @classmethod
    def export_filtered_details(
        cls,
        orig_model,
        filtered_model,
        filter_report,
        vocab_filters,
        export_dir,
    ):
        if filter_report:
            report_file_path = join(export_dir, "_filter_report.csv")
            with open(report_file_path, "w") as report_csvfile:
                csvwriter = writer(report_csvfile)
                header = ["Function", "Pipes", "Args", "Token", "Attributes"]
                filter_report = [header] + [
                    row for row in filter_report if row
                ]
                csvwriter.writerows(filter_report)

        filter_details_file_path = join(export_dir, "_filter_details.txt")
        if exists(filter_details_file_path):
            return
        orig_vocab = len(orig_model.index2word)
        filt_vocab = len(filtered_model.index2word)
        reduction_percent = ((orig_vocab - filt_vocab) / orig_vocab) * 100

        filters_str = "\n"
        function_filters = list(filter(callable, vocab_filters))
        if function_filters:
            fn_filters_str = "\n\n".join(
                map(cls.filters_as_str, function_filters)
            )
            filters_str += "\nFunction Filters: \n{0}".format(fn_filters_str)

        list_filters = list(
            filter(lambda cond: not callable(cond), vocab_filters)
        )
        if list_filters:
            list_filters_str = "\n\n".join(
                map(cls.filters_as_str, list_filters)
            )
            filters_str += "\nList Filters: \n{0}".format(list_filters_str)

        details_str = """
Filtered Vocab Reduction: {filtered_vocab_size}/{original_vocab_size} (-{percentage:.2f}%),\n
Filters: {filters_str}""".format(
            filtered_vocab_size=filt_vocab,
            original_vocab_size=orig_vocab,
            percentage=reduction_percent,
            filters_str=filters_str,
        )
        with open(filter_details_file_path, "w") as filter_details_file:
            filter_details_file.write(details_str)

        return filters_str, reduction_percent

    @classmethod
    def load_gensim_model(cls, source, vocab_filters, data_dir):
        save_model_file = join(data_dir, "_gensim_model.bin")
        if exists(save_model_file):
            filter_details_file = join(data_dir, "_filter_details")
            filter_details = (
                unpickle_file(filter_details_file)
                if exists(filter_details_file)
                else None
            )
            return KeyedVectors.load(save_model_file), filter_details
        source_model_dir = join(dirname(data_dir), source)
        source_model_file = join(source_model_dir, "_gensim_model.bin")
        if exists(source_model_file):
            gensim_model = KeyedVectors.load(source_model_file)
        else:
            try:
                gensim_model = gensim_data.load(source)
                cls.export_gensim_model(gensim_model, data_dir)
            except:
                raise ValueError("Invalid source {0}".format(source))
        if not vocab_filters:
            return gensim_model, None
        filtered_model, filter_details = cls.filter_gensim_model(
            gensim_model, vocab_filters, data_dir
        )
        pickle_file(
            path=join(data_dir, "_filter_details"), data=filter_details
        )
        cls.export_gensim_model(filtered_model, data_dir)
        return filtered_model, filter_details
