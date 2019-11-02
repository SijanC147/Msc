# pylint: disable=unused-argument
from os.path import join, split as path_split
from os import makedirs
from functools import wraps, partial
from pprint import pformat
import tensorflow as tf
from tensorflow.estimator import ModeKeys  # pylint: disable=E0401
from tensorflow.saved_model.signature_constants import (  # pylint: disable=E0401
    DEFAULT_SERVING_SIGNATURE_DEF_KEY,
)
from tensorflow.contrib.estimator import (  # pylint: disable=E0611
    stop_if_no_decrease_hook,
    stop_if_no_increase_hook,
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
    ConsoleLoggerHook,
    EpochCheckpointListener,
)
from tsaplay.utils.tf import streaming_f1_scores, streaming_conf_matrix
from tsaplay.utils.io import (
    extract_config_subset,
    resolve_frequency_steps,
    resolve_summary_step_freq,
    str_snippet,
    cprnt,
)
from tsaplay.constants import SAVE_SUMMARY_STEPS


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


# pylint: disable=too-many-locals
def early_stopping(model, features, labels, spec, params):
    train_hooks = list(spec.training_hooks) or []
    eval_dir = model.estimator.eval_dir()
    makedirs(eval_dir, exist_ok=True)
    config = extract_config_subset(
        config_objs=[params, model.aux_config],
        keywords=["ea", "early_stopping"],
    )
    metric = config.get("metric", "loss")
    comparison = (
        "decrease"
        if metric == "loss"
        else "increase"
        if metric in ["accuracy", "macro-f1", "micro-f1", "weighted-f1"]
        else config.get("comparison")
    )
    epochs = params.get("epochs")
    steps = params.get("steps")
    run_every_steps = config.get(
        "run_every_steps", params.get("epoch_steps") or None
    )
    run_every_secs = config.get("run_every_secs", 60)
    patience = config.get("patience", 10 if epochs is not None else 1000)
    minimum_iter = config.get("minimum_iter", 0)
    if spec.mode == ModeKeys.TRAIN:
        cprnt(
            INFO=(
                "INFO Early Stopping: \n"
                + "\t".join(
                    [
                        "metric: {metric}",
                        "mode: {comparison}",
                        "run every: {run_every_steps}",
                        "patience: {patience}",
                        "minimum: {minimum_iter}",
                        "maximum: {maximum_iter}",
                    ]
                )
            ).format(
                metric=metric,
                comparison=comparison,
                run_every_steps=(
                    "1 epoch ({} steps)" if epochs is not None else "{} steps"
                ).format(run_every_steps)
                if run_every_steps
                else "{} secs".format(run_every_secs),
                patience=(
                    "{} epoch(s)" if epochs is not None else "{} steps"
                ).format(patience),
                minimum_iter=(
                    "{} epoch(s)" if epochs is not None else "{} steps"
                ).format(minimum_iter)
                if minimum_iter > 0
                else "Indefinite",
                maximum_iter=(
                    "{} epoch(s)" if epochs is not None else "{} steps"
                ).format(epochs if epochs is not None else steps),
            )
        )
    early_stopping_hook_fn = (
        stop_if_no_increase_hook
        if comparison == "increase"
        else stop_if_no_decrease_hook
    )
    early_stopping_hook_args = {
        "estimator": model.estimator,
        "eval_dir": eval_dir,
        "metric_name": metric,
        "min_steps": (
            minimum_iter
            * (params.get("epoch_steps") if epochs is not None else 1)
        ),
        "run_every_steps": run_every_steps,
        "run_every_secs": (
            None if run_every_steps is not None else run_every_secs
        ),
        ("max_steps_without_{}".format(comparison)): (
            patience * (params.get("epoch_steps") if epochs is not None else 1)
        ),
    }
    train_hooks += [early_stopping_hook_fn(**early_stopping_hook_args)]
    return spec._replace(training_hooks=train_hooks)


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
            epoch_steps=params.get("epoch_steps"),
            n_hops=params.get("n_hops"),
            n_picks=model.aux_config.get("attn_heatmaps_n", 1),
            freq=model.aux_config.get("attn_heatmaps_freq", 10),
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
            epoch_steps=params.get("epoch_steps"),
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
def checkpoint_saver(model, features, labels, spec, params):
    train_hooks = list(spec.training_hooks) or []
    checkpoint_listeners = (
        [
            EpochCheckpointListener(
                model_dir=model.run_config.model_dir,
                epoch_steps=params["epoch_steps"],
            )
        ]
        if params.get("epochs") is not None
        else []
    )
    train_hooks += [
        tf.train.CheckpointSaverHook(
            model.run_config.model_dir,
            save_steps=params["epoch_steps"],
            listeners=checkpoint_listeners,
            scaffold=spec.scaffold,
        )
    ]
    return spec._replace(training_hooks=train_hooks)


@only(["TRAIN", "EVAL"])
def logging(model, features, labels, spec, params):
    config = extract_config_subset(
        config_objs=[params, model.aux_config], keywords="logging"
    )
    id_tag = str_snippet(
        params.get("contd_tag")
        if params.get("contd_tag") is not None
        else path_split(model.run_config.model_dir)[1]
    )
    if spec.mode == ModeKeys.TRAIN:
        cprnt(
            tf=True,
            INFO=(
                "\n".join(
                    [
                        "INFO Run Configuration:",
                        pformat(model.run_config.__dict__),
                        "INFO AUX Configuration:",
                        pformat(model.aux_config),
                        "INFO Parameters:",
                        pformat(model.params),
                    ]
                )
            ),
        )
        std_metrics = {
            "accuracy": tf.metrics.accuracy(
                labels=labels,
                predictions=spec.predictions["class_ids"],
                name="acc_op",
            )
        }
        tensors_to_log = {
            "step": tf.train.get_global_step(),
            "loss": spec.loss,
            "accuracy": std_metrics["accuracy"][1],
        }
        console_template = config.get(
            "train_template",
            id_tag
            + "TRAIN \t STEP: {step} \t EPOCH: {epoch:.1f} \t|\t"
            + "acc: {accuracy:.5f} \t loss: {loss:.8f} |\t "
            + "duration: {duration:.2f}s"
            + "sec/step: {sec_per_step:.2f}s step/sec: {step_per_sec:.2f}",
        )
        # summary_freq = resolve_frequency_steps(
        #     freq=config.get("summary_freq"),
        #     epochs=params.get("epochs"),
        #     epoch_steps=params["epoch_steps"],
        #     default=SAVE_SUMMARY_STEPS,
        # )
        summary_step_freq = resolve_summary_step_freq(
            config=config,
            epochs=params.get("epochs"),
            epoch_steps=params["epoch_steps"],
        )
        train_hooks = list(spec.training_hooks) or []
        train_hooks += [
            ConsoleLoggerHook(
                mode=ModeKeys.TRAIN,
                epoch_steps=params["epoch_steps"],
                every_n_iter=config.get("every_n_iter"),
                summary_writer=tf.summary.FileWriterCache.get(
                    model.run_config.model_dir
                ),
                summary_freq=summary_step_freq,
                tensors={**tensors_to_log, "summary": tf.summary.merge_all()},
                template=console_template,
            )
        ]
        spec = spec._replace(training_hooks=train_hooks)
    elif spec.mode == ModeKeys.EVAL:
        eval_hooks = list(spec.evaluation_hooks) or []
        eval_hooks += [
            ConsoleLoggerHook(
                mode=ModeKeys.EVAL,
                epoch_steps=params["epoch_steps"],
                tensors={k: v[0] for k, v in spec.eval_metric_ops.items()},
                template=(
                    id_tag
                    + "EVAL \t STEP: {step} \t EPOCH: {epoch:.1f} \t|\t"
                    + "acc: {accuracy:.5f} \t mpc_acc: {mpc_accuracy:.5f} \t"
                    + "macro-f1: {macro-f1:.5f} \t"
                    + "weighted-f1: {weighted-f1:.5f}"
                    + "\n{conf-mat}"
                ),
            )
        ]
        spec = spec._replace(evaluation_hooks=eval_hooks)

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
        tf.summary.scalar("loss", spec.loss)
        tf.summary.scalar("accuracy", std_metrics["accuracy"][1])
    return spec


@only(["TRAIN", "EVAL"])
def metadata(model, features, labels, spec, params):
    # summary_steps = (
    #     model.run_config.save_summary_steps or params["epoch_steps"]
    # )
    summary_step_freq = resolve_summary_step_freq(
        config_objs=[params, model.aux_config],
        keywords=["logging", "metadata"],
        epochs=params.get("epochs"),
        epoch_steps=params["epoch_steps"],
    )
    if spec.mode == ModeKeys.TRAIN:
        train_hooks = list(spec.training_hooks) or []
        metadata_dir = model.run_config.model_dir
        makedirs(metadata_dir, exist_ok=True)
        train_hooks += [
            MetadataHook(
                summary_writer=tf.summary.FileWriterCache.get(
                    model.run_config.model_dir
                ),
                save_steps=summary_step_freq,
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


@only(["TRAIN"])
def histograms(model, features, labels, spec, params):
    # config = extract_config_subset(
    #     config_objs=[params, model.aux_config],
    #     keywords=["logging", "histograms"],
    # )
    # summary_freq = resolve_frequency_steps(
    #     freq=config.get("summary_freq"),
    #     epochs=params.get("epochs"),
    #     epoch_steps=params["epoch_steps"],
    #     default=SAVE_SUMMARY_STEPS,
    # )
    summary_step_freq = resolve_summary_step_freq(
        config_objs=[params, model.aux_config],
        keywords=["logging", "histograms"],
        epochs=params.get("epochs"),
        epoch_steps=params["epoch_steps"],
    )
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
                every_n_iter=summary_step_freq,
            )
        ]
        spec = spec._replace(training_hooks=train_hooks)
    return spec


@only(["TRAIN"])
def timeline(model, features, labels, spec, params):
    # config = extract_config_subset(
    #     config_objs=[params, model.aux_config],
    #     keywords=["logging", "timeline"],
    # )
    summary_step_freq = resolve_summary_step_freq(
        config_objs=[params, model.aux_config],
        keywords=["logging", "timeline"],
        epochs=params.get("epochs"),
        epoch_steps=params["epoch_steps"],
    )
    train_hooks = list(spec.training_hooks) or []
    profiler_dir = join(model.run_config.model_dir)
    makedirs(profiler_dir, exist_ok=True)
    train_hooks += [
        tf.train.ProfilerHook(
            save_steps=summary_step_freq,
            output_dir=profiler_dir,
            show_memory=True,
        )
    ]
    return spec._replace(training_hooks=train_hooks)
