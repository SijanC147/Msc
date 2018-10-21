import time
import inspect
from os import environ
from datetime import timedelta
from functools import wraps, partial
import tensorflow as tf
from tensorflow.estimator import ModeKeys  # pylint: disable=E0401
from tsaplay.utils.io import cprnt, comet_pretty_log
from tsaplay.utils.data import prep_dataset
from tsaplay.utils.tf import scaffold_init_fn_on_spec


def timeit(pre="", post=""):
    def inner_decorator(func):
        @wraps(func)
        def wrapper(*args, **kw):
            if environ.get("TIMEIT", "ON") == "ON":
                name = func.__qualname__ + "():"
                cprnt(r=name, g=pre)
                ts = time.time()
                result = func(*args, **kw)
                te = time.time()
                time_taken = timedelta(seconds=(te - ts))
                cprnt(r=name, g=post + " in", row=str(time_taken))
                return result
            return func(*args, **kw)

        return wrapper

    return inner_decorator


def embed_sequences(model_fn):
    @wraps(model_fn)
    def wrapper(self, features, labels, mode, params):
        vocab_size = params["_vocab_size"]
        dim_size = params["_embedding_dim"]
        embedding_init = params["_embedding_init"]
        trainable = params.get("train_embeddings", True)
        with tf.variable_scope("embedding_layer", reuse=tf.AUTO_REUSE):
            embeddings = tf.get_variable(
                "embeddings",
                shape=[vocab_size, dim_size],
                initializer=embedding_init,
                partitioner=tf.fixed_size_partitioner(num_shards=6),
                trainable=trainable,
                dtype=tf.float32,
            )

        embedded_sequences = {}
        for key, value in features.items():
            if "_ids" in key:
                component = key.replace("_ids", "")
                embdd_key = component + "_emb"
                # embedded_sequence = tf.contrib.layers.embed_sequence(
                #     ids=value,
                #     initializer=embeddings,
                #     scope="embedding_layer",
                #     reuse=True,
                # )
                embedded_sequence = tf.nn.embedding_lookup(
                    params=embeddings, ids=value
                )
                embedded_sequences[embdd_key] = embedded_sequence
        features.update(embedded_sequences)
        spec = model_fn(self, features, labels, mode, params)

        # def init_embeddings(sess):
        #     value = embedding_init()
        #     sess.run(embeddings.initializer, {embeddings.initial_value: value})
        # spec = scaffold_init_fn_on_spec(spec, init_embeddings)
        return spec

    return wrapper


def sharded_saver(model_fn):
    @wraps(model_fn)
    def wrapper(self, features, labels, mode, params):
        spec = model_fn(self, features, labels, mode, params)
        scaffold = spec.scaffold or tf.train.Scaffold()
        scaffold._saver = tf.train.Saver(sharded=True)
        return spec._replace(scaffold=scaffold)

    return wrapper


def cometml(model_fn):
    @wraps(model_fn)
    def wrapper(self, features, labels, mode, params):
        comet = self.comet_experiment
        if comet is not None and mode in [ModeKeys.TRAIN, ModeKeys.EVAL]:
            run_config = self.run_config.__dict__
            comet_pretty_log(comet, self.aux_config, prefix="AUX")
            comet_pretty_log(comet, run_config, prefix="RUNCONFIG")
            comet_pretty_log(comet, params, hparams=True)
            comet.set_code(inspect.getsource(self.__class__))
            comet.set_filename(inspect.getfile(self.__class__))
            if mode == ModeKeys.TRAIN:
                global_step = tf.train.get_global_step()
                comet.set_step(global_step)
                comet.log_current_epoch(global_step)

            spec = model_fn(self, features, labels, mode, params)

            if mode == ModeKeys.TRAIN:

                def export_graph_to_comet(sess):
                    comet.set_model_graph(sess.graph)

                spec = scaffold_init_fn_on_spec(spec, export_graph_to_comet)

                comet.log_epoch_end(global_step)
            return spec
        return model_fn(self, features, labels, mode, params)

    return wrapper


def attach(addons, modes=None, order="POST"):
    def decorator(model_fn):
        @wraps(model_fn)
        def wrapper(self, features, labels, mode, params):
            aux = self.aux_config
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
                if mode in target_modes and aux.get(addon.__name__, True)
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
            if mode in ["TRAIN", "EVAL"]:
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

                return prep_dataset(
                    tfrecords=tfrecords,
                    params=params,
                    processing_fn=process_dataset,
                    mode=mode,
                )

            raise ValueError("Invalid mode: {0}".format(mode))

        return input_fn

    return decorator


def wrap_parsing_fn(parsing_fn):
    @wraps(parsing_fn)
    def wrapper(path):
        try:
            sentences, targets, offsets, labels = parsing_fn(path)
            return {
                "sentences": sentences,
                "targets": targets,
                "offsets": offsets,
                "labels": labels,
            }
        except ValueError:
            sentences, targets, labels = parsing_fn(path)
            offsets = [
                sentence.lower().find(target.lower())
                for sentence, target in zip(sentences, targets)
            ]
            return {
                "sentences": sentences,
                "targets": targets,
                "offsets": offsets,
                "labels": labels,
            }

    return wrapper

