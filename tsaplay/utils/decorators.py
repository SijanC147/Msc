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


def attach(target_modes, addons):
    def decorator(model_fn):
        @wraps(model_fn)
        def wrapper(self, features, labels, mode, params):
            targets = [
                {
                    "TRAIN": ModeKeys.TRAIN,
                    "EVAL": ModeKeys.EVAL,
                    "PREDICT": ModeKeys.PREDICT,
                }.get(trg_mode)
                for trg_mode in target_modes
            ]
            if self.comet_experiment is not None:
                self.comet_experiment.context = mode
                self.comet_experiment.log_multiple_params(params)
                self.comet_experiment.set_step(tf.train.get_global_step())
                self.comet_experiment.set_code(
                    inspect.getsource(self.__class__)
                )
                self.comet_experiment.set_filename(
                    inspect.getfile(self.__class__)
                )
            spec = model_fn(self, features, labels, mode, params)
            if mode in targets:
                for add_on in addons:
                    spec = add_on(self, spec, features, labels, params)
            return spec

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
