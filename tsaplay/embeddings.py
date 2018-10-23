from os import makedirs
from os.path import join, exists
from math import sqrt, floor
from inspect import getsource, Parameter, signature
from hashlib import md5
from warnings import warn
import tensorflow as tf
import numpy as np
import gensim.downloader as gensim_data
from gensim.models import KeyedVectors
import spacy
from spacy.language import Language
from spacy.tokens import Doc
from tsaplay.constants import EMBEDDING_DATA_PATH, SPACY_MODEL
from tsaplay.utils.decorators import timeit

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


class Embedding:
    def __init__(
        self,
        source,
        oov=None,
        num_shards=None,
        vocab_filters=None,
        data_root=None,
    ):
        self._data_root = data_root or EMBEDDING_DATA_PATH
        self._oov = oov or self.default_oov
        if not callable(self._oov):
            warn("OOV parameter is not callable, falling back to default.")
            self._oov = self.default_oov
        self._source = source
        self._vocab_filters = vocab_filters or []
        if self._vocab_filters:
            filters_list = [self._source] + self._vocab_filters
            filters_list_str = map(self.filters_as_str, filters_list)
            filters_list_md5 = [
                md5(str(filt).encode("utf-8")).hexdigest()
                for filt in filters_list_str
            ].sort()
            self._filter_hash_id = md5(
                str(filters_list_md5).encode("utf-8")
            ).hexdigest()
        else:
            self._filter_hash_id = None
        self._gen_dir = join(self._data_root, self.name)
        makedirs(self._gen_dir, exist_ok=True)
        self._gensim_model = self._load_gensim_model(
            self._source, self._vocab_filters
        )
        self._flags = {
            "<PAD>": np.zeros(shape=self.dim_size),
            "<OOV>": self._oov(size=self.dim_size),
        }
        self._vectors = np.concatenate(
            [
                [self._flags["<PAD>"]],
                [self._flags["<OOV>"]],
                self._gensim_model.vectors,
            ]
        ).astype(np.float32)
        self._num_shards = num_shards or self.smallest_partition_divisor(
            self.vocab_size
        )
        if self.vocab_size % self._num_shards != 0:
            self._num_shards = self.smallest_partition_divisor(self.vocab_size)
            warn(
                "vocab {0} not a multiple of {1}, using {2} shards".format(
                    self.vocab_size, num_shards, self._num_shards
                )
            )
        self._vocab_file_path = self.export_vocab_files(
            self.vocab, self.gen_dir
        )

    @property
    def source(self):
        return self._source

    @property
    def name(self):
        if not self._filter_hash_id:
            return self._source
        return "--".join([self._source, self._filter_hash_id])

    @property
    def oov(self):
        return self._oov

    @property
    def gen_dir(self):
        return self._gen_dir

    @property
    def vocab(self):
        return [*self.flags] + self._gensim_model.index2word

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
    def flags(self):
        return self._flags

    @property
    def vectors(self):
        return self._vectors

    @property
    def num_shards(self):
        return self._num_shards

    @property
    def initializer(self, structure=None):
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
        return np.random.uniform(low=-0.03, high=0.03, size=size)

    @classmethod
    def filters_as_str(cls, filter_condition):
        if callable(filter_condition):
            return [getsource(filter_condition)]
        return list(filter_condition)

    @classmethod
    def smallest_partition_divisor(cls, vocab_size):
        if vocab_size % 2 == 0:
            return 2
        square_root = floor(sqrt(vocab_size))
        for i in (3, square_root, 2):
            if vocab_size % i == 0:
                return i
            return 1

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
    def filter_gensim_model(cls, gensim_model, vocab_filters):
        entities = gensim_model.index2word

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
            required_pipes = set(sum(required_pipes, []))

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
                            return keep
                    return keep

                doc = embedding_lang(entities)
                entities = [token.text for token in filter(grouped, doc)]

        list_filters = filter(lambda cond: not callable(cond), vocab_filters)
        list_filters = sum(list_filters, [])
        if list_filters:
            entities = list(set(entities) & set(list_filters))

        weights = [gensim_model.get_vector(entity) for entity in entities]
        filtered_model = KeyedVectors(gensim_model.vector_size)
        filtered_model.add(entities, weights)
        return filtered_model

    @classmethod
    def export_filtered_details(
        cls, orig_model, filtered_model, vocab_filters, export_dir
    ):
        filter_details_file_path = join(export_dir, "_filter_details.txt")
        if exists(filter_details_file_path):
            return
        orig_vocab = len(orig_model.index2word)
        filt_vocab = len(filtered_model.index2word)
        reduction_percent = ((orig_vocab - filt_vocab) / orig_vocab) * 100

        def pprint_filters(filt):
            filt_str = cls.filters_as_str(filt)
            if not callable(filt):
                return str(filt_str)
            return filt_str[0]

        filters_str = ""
        function_filters = list(filter(callable, vocab_filters))
        if function_filters:
            fn_filters_str = "\n\n".join(map(pprint_filters, function_filters))
            filters_str += "Function Filters: {0}".format(fn_filters_str)

        list_filters = filter(lambda cond: not callable(cond), vocab_filters)
        if list_filters:
            list_filters_str = "\n\n".join(map(pprint_filters, list_filters))
            filters_str += "List Filters: {0}".format(list_filters_str)

        details_str = """
Filtered Vocab Reduction: {filtered_vocab_size}/{original_vocab_size} ({percentage:.2f}%),
Filters: {filters_str}""".format(
            filtered_vocab_size=filt_vocab,
            original_vocab_size=orig_vocab,
            percentage=reduction_percent,
            filters_str=filters_str,
        )
        with open(filter_details_file_path, "w") as filter_details_file:
            filter_details_file.write(details_str)

    def _load_gensim_model(self, source, vocab_filters):
        save_model_file = join(self.gen_dir, "_gensim_model.bin")
        if exists(save_model_file):
            return KeyedVectors.load(save_model_file)
        source_model_dir = join(self._data_root, self._source)
        source_model_file = join(source_model_dir, "_gensim_model.bin")
        if exists(save_model_file):
            gensim_model = KeyedVectors.load(source_model_file)
        else:
            try:
                gensim_model = gensim_data.load(source)
            except:
                raise ValueError("Invalid source {0}".format(source))
        if not vocab_filters:
            self.export_gensim_model(gensim_model, self.gen_dir)
            return gensim_model
        self.export_gensim_model(gensim_model, source_model_dir)
        filtered_model = self.filter_gensim_model(gensim_model, vocab_filters)
        self.export_gensim_model(filtered_model, self.gen_dir)
        self.export_filtered_details(
            gensim_model, filtered_model, vocab_filters, self.gen_dir
        )
        return filtered_model
