# pylint: disable=invalid-name,unused-argument
from functools import partial


def _no_pipe_filter(token, pipes=[None], attrs=[]):
    return True in [
        getattr(token, attr)
        if attr[0] != "!"
        else not (getattr(token, attr[1:]))
        for attr in attrs
    ]


def _pipe_filter(token, pipes=[], tags=[], attr=""):
    keep = True
    include = [tag for tag in tags if tag[0] != "!"]
    if include:
        keep = getattr(token, attr) in include
        if not keep:
            return keep
    exclude = [tag[1:] for tag in tags if tag[0] == "!"]
    if exclude:
        return getattr(token, attr) not in exclude
    return keep


_pos_pipe_filter = partial(_pipe_filter, pipes=["pos"], attr="pos_")
_dep_pipe_filter = partial(_pipe_filter, pipes=["dep"], attr="dep_")
_ner_pipe_filter = partial(_pipe_filter, pipes=["ner"], attr="ent_type")


no_numbers = partial(_no_pipe_filter, attrs=["!like_num", "is_alpha"])
no_urls = partial(_no_pipe_filter, attrs=["!like_url"])
no_emails = partial(_no_pipe_filter, attrs=["!like_email"])
no_currency = partial(_no_pipe_filter, attrs=["!is_currency"])
only_ascii = partial(_no_pipe_filter, attrs=["is_ascii"])


no_unknown_pos = partial(_pos_pipe_filter, tags=["!X", "!XX"])
no_proper_nouns = partial(_pos_pipe_filter, tags=["!PROPN"])

pos_set_one = partial(_pos_pipe_filter, tags=["ADJ", "ADV", "NOUN", "VERB"])
