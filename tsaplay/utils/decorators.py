import time
import inspect
import tensorflow as tf
from contextlib import suppress
from datetime import timedelta
from functools import wraps, partial
from tsaplay.utils.io import cprnt
from tsaplay.utils.data import parse_tf_example, prep_dataset
from tensorflow.estimator import RunConfig, Estimator  # pylint: disable=E0401
from tensorflow.estimator import (  # pylint: disable=E0401
    ModeKeys,
    RunConfig,
    Estimator,
)
from tensorflow.saved_model.signature_constants import (
    DEFAULT_SERVING_SIGNATURE_DEF_KEY
)
from tensorflow.estimator.export import (  # pylint: disable=E0401
    PredictOutput,
    RegressionOutput,
    ClassificationOutput,
    ServingInputReceiver,
)


def timeit(pre="", post=""):
    def inner_decorator(func):
        @wraps(func)
        def wrapper(*args, **kw):
            name = func.__qualname__ + "():"
            cprnt(r=name, g=pre)
            ts = time.time()
            result = func(*args, **kw)
            te = time.time()
            time_taken = timedelta(seconds=(te - ts))
            cprnt(r=name, g=post + " in", row=str(time_taken))
            return result

        return wrapper

    return inner_decorator


def initialize_estimator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        emb_params = (
            kwargs.get("embedding_params")
            or kwargs.get("feature_provider").embedding_params
        )
        args[0].params = {**args[0].params, **emb_params}
        args[0]._estimator = Estimator(
            model_fn=args[0]._model_fn,
            params=args[0].params,
            config=args[0].run_config,
        )
        return func(*args, **kwargs)

    return wrapper


def attach(addons, modes=None, order="POST"):
    def decorator(model_fn):
        @wraps(model_fn)
        def wrapper(self, features, labels, mode, params):
            targets = modes or ["train", "eval", "predict"]
            target_modes = [
                {
                    "train": ModeKeys.TRAIN,
                    "eval": ModeKeys.EVAL,
                    "predict": ModeKeys.PREDICT,
                }.get(trg_mode.lower())
                for trg_mode in targets
            ]
            applicable_addons = [
                addon
                for addon in addons
                if mode in target_modes and params.get(addon.__name__, True)
            ]
            if order == "PRE":
                for add_on in applicable_addons:
                    add_on(self, features, labels, mode, params)
                spec = model_fn(self, features, labels, mode, params)
            elif order == "POST":
                spec = model_fn(self, features, labels, mode, params)
                for add_on in applicable_addons:
                    spec = add_on(self, features, labels, spec, params)
            return spec

        return wrapper

    return decorator


addon = partial(attach, order="POST")
prepare = partial(attach, order="PRE")


def only(modes):
    def decorator(addon_fn):
        @wraps(addon_fn)
        def wrapper(*args, **kwargs):
            try:
                context_mode = args[3].mode
            except AttributeError:
                context_mode = args[3]
            except IndexError:
                context_spec = kwargs.get("spec")
                if context_spec is not None:
                    context_mode = context_spec.mode
                else:
                    context_mode = kwargs.get("mode")
            if context_mode.lower() in [m.lower() for m in modes]:
                return addon_fn(*args, **kwargs)
            return

        return wrapper

    return decorator


def make_input_fn(mode):
    def decorator(func):
        @wraps(func)
        def input_fn(*args, **kwargs):
            if mode == "TRAIN" or mode == "EVAL":
                try:
                    tfrecord = args[1]
                except IndexError:
                    tfrecord = kwargs.get("tfrecord")
                try:
                    batch_size = args[2]
                except IndexError:
                    batch_size = kwargs.get("batch_size")

                def process_dataset(features, labels):
                    try:
                        return (args[0].processing_fn(features), labels)
                    except AttributeError:
                        return (features, labels)

                dataset = prep_dataset(
                    tfrecord=tfrecord,
                    batch_size=batch_size,
                    processing_fn=process_dataset,
                    mode=mode,
                )

                return dataset.make_one_shot_iterator().get_next()
            else:
                raise ValueError("Invalid mode: {0}".format(mode))

        return input_fn

    return decorator
