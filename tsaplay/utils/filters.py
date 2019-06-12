# pylint: disable=invalid-name,unused-argument
from functools import partial
from inspect import signature, getmembers
from tsaplay.constants import DELIMITER


def default_token_filter(token):
    if token.like_url:
        return False
    if token.like_email:
        return False
    if token.text in ["\uFE0F"]:
        return False
    return True


def group_filter_fns(*filter_fns, stdout_report=False):
    pipes_params = [signature(fn).parameters.get("pipes") for fn in filter_fns]
    pipes_params = [param.default for param in pipes_params if param]
    req_pipes = list(set(sum(pipes_params, [])))

    def _filter_fn(token):
        keep = True
        for filter_fn in filter_fns:
            keep = keep and filter_fn(token)
            if not keep:
                if stdout_report:
                    details = filter_fn_details(filter_fn, token)
                    print(DELIMITER.join(details))
                return keep
        return keep

    report_fields = (
        ["Function", "Pipes", "Args", "Token", "Attributes"]
        if stdout_report
        else None
    )

    return _filter_fn, req_pipes, report_fields


def filter_fn_details(filter_fn, token=None):
    params = signature(filter_fn).parameters
    pipes = params.get("pipes").default if params.get("pipes") else []
    attrs = params.get("attrs").default if not pipes else params["tags"]
    details = [filter_fn.func.__name__, pipes, attrs]
    if token:
        relevant_token_attrs = (
            [attr.replace("!", "") for attr in attrs]
            if not pipes
            else [
                {"pos": "pos_", "ner": "ent_type", "dep": "dep_"}.get(pipe)
                for pipe in pipes
            ]
        )
        token_attrs = getmembers(
            token,
            predicate=lambda member: isinstance(
                member, (str, float, int, bool)
            ),
        )
        token_attrs = [
            "{0}:{1}".format(n, v)
            for n, v in token_attrs
            if n in relevant_token_attrs
        ]
        details += [token.text, token_attrs]
    details = list(map(str, details))
    return details


def filters_registry(key):
    return {
        "no_numbers": no_numbers,
        "no_urls": no_urls,
        "no_emails": no_emails,
        "no_currency": no_currency,
        "only_ascii": only_ascii,
        "no_unknown_pos": no_unknown_pos,
        "no_proper_nouns": no_proper_nouns,
        "pos_set_one": pos_set_one,
        "only_adjectives": only_adjectives,
    }.get(key, False)


def _no_pipe_filter(token, attrs=None):
    return True in [
        getattr(token, attr)
        if attr[0] != "!"
        else not (getattr(token, attr[1:]))
        for attr in attrs
    ]


def _pipe_filter(token, pipes=None, tags=None, attr=""):
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
only_adjectives = partial(_pos_pipe_filter, tags=["ADJ"])
