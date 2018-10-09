import time
import json
import inspect
from datetime import timedelta
from functools import wraps, partial
import tensorflow as tf
from tensorflow.estimator import ModeKeys, Estimator  # pylint: disable=E0401
from tsaplay.utils.io import cprnt
from tsaplay.utils.data import prep_dataset, make_dense_features
from tsaplay.utils.tf import (
    sparse_sequences_to_dense,
    get_seq_lengths,
    scaffold_init_fn_on_spec,
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


def embed_sequences(model_fn):
    @wraps(model_fn)
    def wrapper(self, features, labels, mode, params):
        vocab_size = params["vocab-size"]
        dim_size = params["embedding-dim"]
        embedding_init = params["embedding-init"]
        trainable = params.get("train_embeddings", True)
        with tf.variable_scope("embedding_layer", reuse=tf.AUTO_REUSE):
            embeddings = tf.get_variable(
                "embeddings",
                shape=[vocab_size, dim_size],
                initializer=embedding_init,
                trainable=trainable,
                dtype=tf.float32,
            )
        embedded_sequences = {}
        for key, value in features.items():
            if "_ids" in key:
                component = key.replace("_ids", "")
                embdd_key = component + "_emb"
                embedded_sequence = tf.contrib.layers.embed_sequence(
                    ids=value,
                    initializer=embeddings,
                    scope="embedding_layer",
                    reuse=True,
                )
                embedded_sequences[embdd_key] = embedded_sequence
        features.update(embedded_sequences)
        return model_fn(self, features, labels, mode, params)

    return wrapper


def cometml(model_fn):
    @wraps(model_fn)
    def wrapper(self, features, labels, mode, params):
        if self.comet_experiment is None or mode == ModeKeys.PREDICT:
            return model_fn(self, features, labels, mode, params)
        if mode in [ModeKeys.TRAIN, ModeKeys.EVAL]:
            self.comet_experiment.log_other(
                "Feature Provider", params["feature-provider"]
            )
            self.comet_experiment.log_other(
                "Hidden Units", params["hidden_units"]
            )
            self.comet_experiment.log_other("Batch Size", params["batch-size"])
            run_config = self.run_config.__dict__
            for key, value in run_config.items():
                try:
                    json.dumps(value)
                except TypeError:
                    value = str(value)
                self.comet_experiment.log_other(key, value)
            params_log_other = [
                "model_dir",
                "n_out_classes",
                "vocab-file",
                "embedding-dim",
                "embedding-init",
                "n_attn_heatmaps",
            ]
            for key, value in params.items():
                try:
                    json.dumps(value)
                except TypeError:
                    value = str(value)
                if key in params_log_other:
                    self.comet_experiment.log_other(key, value)
                else:
                    self.comet_experiment.log_parameter(key, value)
            self.comet_experiment.set_code(inspect.getsource(self.__class__))
            self.comet_experiment.set_filename(inspect.getfile(self.__class__))
            self.comet_experiment.log_dataset_hash((features, labels))
            if mode == ModeKeys.TRAIN:
                global_step = tf.train.get_global_step()
                self.comet_experiment.set_step(global_step)
                self.comet_experiment.log_current_epoch(global_step)

            spec = model_fn(self, features, labels, mode, params)

            if mode == ModeKeys.TRAIN:

                def export_graph_to_comet(scaffold, sess):
                    self.comet_experiment.set_model_graph(sess.graph)

                spec = scaffold_init_fn_on_spec(spec, export_graph_to_comet)

                self.comet_experiment.log_epoch_end(global_step)
            return spec

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
            post = pre = False
            try:
                context_mode = args[3].mode
                post = True
            except AttributeError:
                context_mode = args[3]
                pre = True
            except IndexError:
                context_spec = kwargs.get("spec")
                if context_spec is not None:
                    post = True
                    context_mode = context_spec.mode
                else:
                    context_mode = kwargs.get("mode")
                    pre = True
            if context_mode.lower() in [m.lower() for m in modes]:
                if pre:
                    cprnt(wog=addon_fn.__name__)
                    return addon_fn(*args, **kwargs)
                else:
                    cprnt(wog=addon_fn.__name__)
                    return addon_fn(*args, **kwargs)
            cprnt(wor=addon_fn.__name__)
            if post:
                return args[3]

        return wrapper

    return decorator


def make_input_fn(mode):
    def decorator(func):
        @wraps(func)
        def input_fn(*args, **kwargs):
            if mode == "TRAIN" or mode == "EVAL":
                try:
                    tfrecords = args[1]
                except IndexError:
                    tfrecords = kwargs.get("tfrecords")
                try:
                    params = args[2]
                except IndexError:
                    params = kwargs.get("params")

                def process_dataset(features, labels):
                    return (args[0].processing_fn(features), labels)
                    # processed_features = args[0].processing_fn(features)
                    # return (make_dense_features(processed_features), labels)

                dataset = prep_dataset(
                    tfrecords=tfrecords,
                    params=params,
                    processing_fn=process_dataset,
                    mode=mode,
                )

                return dataset.make_one_shot_iterator().get_next()
            else:
                raise ValueError("Invalid mode: {0}".format(mode))

        return input_fn

    return decorator
