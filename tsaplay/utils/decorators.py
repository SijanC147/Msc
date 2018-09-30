import time
from datetime import timedelta
from functools import wraps
from tsaplay.utils.io import _cprnt


def attach_embedding_params(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        emb_params = (
            kwargs.get("embedding_params")
            or kwargs.get("feature_provider").embedding_params
        )
        args[0].params = {**args[0].params, **emb_params}
        return func(*args, **kwargs)

    return wrapper


def timeit(pre="", post=""):
    def inner_decorator(func):
        @wraps(func)
        def wrapper(*args, **kw):
            name = func.__qualname__ + "():"
            _cprnt(r=name, g=pre)
            ts = time.time()
            result = func(*args, **kw)
            te = time.time()
            time_taken = timedelta(seconds=(te - ts))
            _cprnt(r=name, g=post, row=str(time_taken))
            return result

        return wrapper

    return inner_decorator
