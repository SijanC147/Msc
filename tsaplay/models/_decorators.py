import time
from datetime import timedelta
from functools import wraps
from tsaplay.utils._io import gprint


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


def timeit(method):
    def timed(*args, **kw):
        name = method.__name__.upper()
        gprint("Entering {0}".format(name))
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        time_taken = timedelta(seconds=(te - ts))
        gprint("Exited {0} in {1}".format(name, str(time_taken)))
        return result

    return timed


# @classmethod
# def short_name(cls, string, numchar=20):
#     name = str(string, "utf-8")
#     if len(name) > numchar:
#         name = name[:numchar]
#     return "".join(filter(str.isalpha, name))

# @classmethod
# def print_tensors(cls, left, target, right):
#     template = "{3}:\t {0}\t {1}\t {2}"
#     for i in range(20):
#         gprint(
#             template.format(
#                 left[i].name, target[i].name, right[i].name, i + 1
#             )
#         )
