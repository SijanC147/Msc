# pylint: disable=unused-argument
from os.path import join
from os import makedirs
from functools import wraps, partial
import tensorflow as tf
from tensorflow.estimator import ModeKeys  # pylint: disable=E0401
from tensorflow.saved_model.signature_constants import (  # pylint: disable=E0401
    DEFAULT_SERVING_SIGNATURE_DEF_KEY,
)
from tensorflow.contrib.estimator import (  # pylint: disable=E0611
    stop_if_no_decrease_hook,
)
from tensorflow.estimator.export import (  # pylint: disable=E0401
    PredictOutput,
    ClassificationOutput,
)
from tsaplay.hooks import (
    SaveAttentionWeightVector,
    SaveConfusionMatrix,
    MetadataHook,
    LogHistogramsToComet,
)
from tsaplay.utils.tf import streaming_f1_scores, streaming_conf_matrix
from tsaplay.utils.debug import cprnt


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


addon = partial(attach, order="POST")  # pylint: disable=C0103
prepare = partial(attach, order="PRE")  # pylint: disable=C0103


def only(modes):
    def decorator(addon_fn):
        @wraps(addon_fn)
        def wrapper(*args, **kwargs):
            addon_order = None
            try:
                addon_order, context_mode = (
                    ("POST", args[3].mode)
                    if hasattr(args[3], "mode")
                    else ("PRE", args[3])
                )
            except IndexError:
                addon_order, context_mode = (
                    ("PRE", kwargs.get("mode"))
                    if kwargs.get("mode")
                    else ("POST", kwargs.get("spec").mode)
                )
            if context_mode.lower() in [mode.lower() for mode in modes]:
                return addon_fn(*args, **kwargs)
            if addon_order == "POST":
                return args[3]

        return wrapper

    return decorator


@only(["PREDICT"])
def prediction_outputs(model, features, labels, spec, params):
    probs = spec.predictions["probabilities"]
    class_labels = model.aux_config.get(
        "class_labels", ["Negative", "Neutral", "Positive"]
    )
    classes = tf.constant([class_labels])
    classify_output = ClassificationOutput(classes=classes, scores=probs)
    predict_output = PredictOutput(spec.predictions)
    export_outputs = {
        DEFAULT_SERVING_SIGNATURE_DEF_KEY: classify_output,
        "inspect": predict_output,
    }
    all_export_outputs = spec.export_outputs or {}
    all_export_outputs.update(export_outputs)
    return spec._replace(export_outputs=all_export_outputs)


@only(["EVAL"])
def attn_heatmaps(model, features, labels, spec, params):
    eval_hooks = list(spec.evaluation_hooks) or []
    eval_hooks += [
        SaveAttentionWeightVector(
            labels=labels,
            predictions=spec.predictions["class_ids"],
            targets=features["target"],
            classes=model.aux_config["class_labels"],
            summary_writer=tf.summary.FileWriterCache.get(
                join(model.run_config.model_dir, "eval")
            ),
            comet=model.comet_experiment,
            n_picks=model.aux_config.get("n_attn_heatmaps", 5),
            n_hops=params.get("n_hops"),
        )
    ]
    return spec._replace(evaluation_hooks=eval_hooks)


@only(["EVAL"])
def conf_matrix(model, features, labels, spec, params):
    eval_hooks = list(spec.evaluation_hooks) or []
    eval_metrics = spec.eval_metric_ops or {}
    eval_metrics.update(
        {
            "conf-mat": streaming_conf_matrix(
                labels=labels,
                predictions=spec.predictions["class_ids"],
                num_classes=params["_n_out_classes"],
            )
        }
    )
    eval_hooks += [
        SaveConfusionMatrix(
            class_labels=model.aux_config["class_labels"],
            tensor_name="total_confusion_matrix",
            summary_writer=tf.summary.FileWriterCache.get(
                join(model.run_config.model_dir, "eval")
            ),
            comet=model.comet_experiment,
        )
    ]
    return spec._replace(
        eval_metric_ops=eval_metrics, evaluation_hooks=eval_hooks
    )


@only(["EVAL"])
def f1_scores(model, features, labels, spec, params):
    eval_metrics = spec.eval_metric_ops or {}
    eval_metrics.update(
        streaming_f1_scores(
            labels=labels,
            predictions=spec.predictions["class_ids"],
            num_classes=params["_n_out_classes"],
        )
    )
    return spec._replace(eval_metric_ops=eval_metrics)


@only(["TRAIN"])
def logging(model, features, labels, spec, params):
    std_metrics = {
        "accuracy": tf.metrics.accuracy(
            labels=labels,
            predictions=spec.predictions["class_ids"],
            name="acc_op",
        )
    }
    train_hooks = list(spec.training_hooks) or []
    train_hooks += [
        tf.train.LoggingTensorHook(
            tensors={
                "step": tf.train.get_global_step(),
                "loss": spec.loss,
                "accuracy": std_metrics["accuracy"][1],
            },
            every_n_iter=model.run_config.save_summary_steps,
        )
    ]
    return spec._replace(training_hooks=train_hooks)


@only(["TRAIN"])
def histograms(model, features, labels, spec, params):
    trainables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    names = [variable.name.replace(":", "_") for variable in trainables]
    for (name, variable) in zip(names, trainables):
        tf.summary.histogram(name, variable)
    if model.comet_experiment:
        train_hooks = list(spec.training_hooks) or []
        train_hooks += [
            LogHistogramsToComet(
                comet=model.comet_experiment,
                names=names,
                trainables=trainables,
                every_n_iter=model.run_config.save_summary_steps,
            )
        ]
        spec = spec._replace(training_hooks=train_hooks)
    return spec


@only(["TRAIN", "EVAL"])
def scalars(model, features, labels, spec, params):
    std_metrics = {
        "accuracy": tf.metrics.accuracy(
            labels=labels,
            predictions=spec.predictions["class_ids"],
            name="acc_op",
        ),
        "mpc_accuracy": tf.metrics.mean_per_class_accuracy(
            labels=labels,
            predictions=spec.predictions["class_ids"],
            num_classes=params["_n_out_classes"],
            name="mpc_acc_op",
        ),
    }
    if spec.mode == ModeKeys.EVAL:
        eval_metrics = spec.eval_metric_ops or {}
        eval_metrics.update(std_metrics)
        spec = spec._replace(eval_metric_ops=eval_metrics)
    else:
        tf.summary.scalar("accuracy", std_metrics["accuracy"][1])
    return spec


def early_stopping(model, features, labels, spec, params):
    train_hooks = list(spec.training_hooks) or []
    eval_dir = model.estimator.eval_dir()
    makedirs(eval_dir, exist_ok=True)
    epoch_steps = params.get("epoch_steps")
    patience = params.get("patience", model.aux_config.get("patience"))
    allowance = params.get("allowance", model.aux_config.get("allowance"))
    metric = params.get(
        "early_stopping_metric",
        model.aux_config.get("early_stopping_metric", "loss"),
    )
    train_hooks += [
        stop_if_no_decrease_hook(
            estimator=model.estimator,
            metric_name=metric,
            max_steps_without_decrease=patience * epoch_steps,
            min_steps=allowance * epoch_steps,
            run_every_steps=epoch_steps,
        )
    ]
    return spec._replace(training_hooks=train_hooks)


@only(["TRAIN"])
def timeline(model, features, labels, spec, params):
    train_hooks = list(spec.training_hooks) or []
    profiler_dir = join(model.run_config.model_dir)
    makedirs(profiler_dir, exist_ok=True)
    summary_steps = model.run_config.save_summary_steps
    train_hooks += [
        tf.train.ProfilerHook(
            save_steps=summary_steps, output_dir=profiler_dir, show_memory=True
        )
    ]
    return spec._replace(training_hooks=train_hooks)


@only(["TRAIN", "EVAL"])
def metadata(model, features, labels, spec, params):
    summary_steps = model.run_config.save_summary_steps
    if spec.mode == ModeKeys.TRAIN:
        train_hooks = list(spec.training_hooks) or []
        metadata_dir = model.run_config.model_dir
        makedirs(metadata_dir, exist_ok=True)
        train_hooks += [
            MetadataHook(
                summary_writer=tf.summary.FileWriterCache.get(
                    model.run_config.model_dir
                ),
                save_steps=summary_steps,
            )
        ]
        spec = spec._replace(training_hooks=train_hooks)
    elif spec.mode == ModeKeys.EVAL:
        eval_hooks = list(spec.evaluation_hooks) or []
        metadata_dir = model.estimator.eval_dir()
        makedirs(metadata_dir, exist_ok=True)
        eval_hooks += [
            MetadataHook(
                summary_writer=tf.summary.FileWriterCache.get(
                    join(model.run_config.model_dir, "eval")
                ),
                save_batches="once",
            )
        ]
        spec = spec._replace(evaluation_hooks=eval_hooks)
    return spec
