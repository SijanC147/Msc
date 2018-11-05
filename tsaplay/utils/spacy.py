from os.path import join
from tqdm import tqdm
import spacy
from spacy.language import Language
from spacy.tokens import Doc
from tsaplay.constants import SPACY_MODEL


def pipe_docs(docs, nlp=None, pbar_desc=None, **kwargs):
    disable = kwargs.get("disable", ["parser", "ner"])
    batch_size = kwargs.get("batch_size", 100)
    n_threads = kwargs.get("n_threads", -1)
    nlp = nlp or spacy.load(SPACY_MODEL, disable=disable)
    docs_pipe = nlp.pipe(docs, batch_size=batch_size, n_threads=n_threads)
    if pbar_desc:
        return tqdm(docs_pipe, total=len(docs), desc=pbar_desc)
    return docs_pipe


def word_counts(docs, **kwargs):
    disable = kwargs.get("disable", ["parser", "ner"])
    nlp = kwargs.get("nlp") or spacy.load(SPACY_MODEL, disable=disable)
    docs_pipe = pipe_docs(docs, nlp=nlp, pbar_desc=kwargs.get("pbar_desc"))
    return (
        [
            (nlp.vocab.strings[key], count)
            for (key, count) in doc.count_by(spacy.attrs.ORTH).items()
        ]
        for doc in docs_pipe
    )


def pipe_vocab(vocab, pipes, **kwargs):
    nlp = spacy.load(SPACY_MODEL)
    lang = Language(
        nlp.vocab, make_doc=lambda voc: Doc(nlp.vocab, words=[voc])
    )
    for pipe in pipes:
        comp_dir = {"dep": "parser", "ner": "ner", "pos": "tagger"}.get(pipe)
        if comp_dir:
            comp = (
                spacy.pipeline.DependencyParser(nlp.vocab)
                if pipe == "dep"
                else spacy.pipeline.EntityRecognizer(nlp.vocab)
                if pipe == "ner"
                else spacy.pipeline.Tagger(nlp.vocab)
            )
            comp.from_disk(join(SPACY_MODEL, comp_dir))
            lang.add_pipe(comp)
    lang.max_length = len(vocab) + 1
    docs_pipe = pipe_docs(vocab, lang, pbar_desc=kwargs.get("pbar_desc"))
    return docs_pipe

