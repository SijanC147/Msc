# pylint: disable=invalid-name
from functools import partial


def no_pipe_filter(token, pipes=[None], attrs=[]):
    return True in [
        getattr(token, attr)
        if attr[0] != "!"
        else not (getattr(token, attr[1:]))
        for attr in attrs
    ]


def pipe_filter(token, pipes=[], tags=[], attr=""):
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


no_numbers = partial(no_pipe_filter, attrs=["!like_num", "is_alpha"])
no_urls = partial(no_pipe_filter, attrs=["!like_url"])
no_emails = partial(no_pipe_filter, attrs=["!like_email"])
no_currency = partial(no_pipe_filter, attrs=["!is_currency"])
only_ascii = partial(no_pipe_filter, attrs=["is_ascii"])


pos_pipe_filter = partial(pipe_filter, pipes=["pos"], attr="pos_")
dep_pipe_filter = partial(pipe_filter, pipes=["dep"], attr="dep_")
ner_pipe_filter = partial(pipe_filter, pipes=["ner"], attr="ent_type")


no_unknown_pos = partial(pos_pipe_filter, tags=["!X", "!XX"])
no_proper_nouns = partial(pos_pipe_filter, tags=["!PROPN"])
