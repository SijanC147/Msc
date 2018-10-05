import tensorflow as tf
import inspect
from functools import partial
from os.path import join
from os import makedirs, environ
from tensorflow.estimator import (  # pylint: disable=E0401
    ModeKeys,
    RunConfig,
    Estimator,
)
from tensorflow.saved_model.signature_constants import (
    DEFAULT_SERVING_SIGNATURE_DEF_KEY
)
from tensorflow.contrib.estimator import (  # pylint: disable=E0611
    stop_if_no_decrease_hook
)
from tensorflow.estimator.export import (  # pylint: disable=E0401
    PredictOutput,
    RegressionOutput,
    ClassificationOutput,
    ServingInputReceiver,
)
from tsaplay.utils.io import cprnt
from tsaplay.utils.decorators import only
from tsaplay.hooks.SaveAttentionWeightVector import SaveAttentionWeightVector
from tsaplay.hooks.SaveConfusionMatrix import SaveConfusionMatrix


def embeddings(model, features, labels, mode, params):
    vocab_size = params["vocab_size"]
    dim_size = params["embedding_dim"]
    num_shards = params["num_shards"]
    trainable = params.get("train_embeddings", True)
    with tf.variable_scope("embedding_layer", reuse=tf.AUTO_REUSE):
        tf.get_variable(
            "embeddings",
            shape=[vocab_size, dim_size],
            partitioner=tf.fixed_size_partitioner(num_shards),
            trainable=trainable,
            dtype=tf.float32,
        )


@only(["TRAIN", "EVAL"])
def cometml(model, features, labels, mode, params):
    if model.comet_experiment is not None:
        model.comet_experiment.context = mode
        model.comet_experiment.log_multiple_params(params)
        model.comet_experiment.set_step(tf.train.get_global_step())
        model.comet_experiment.set_code(inspect.getsource(model.__class__))
        model.comet_experiment.set_filename(inspect.getfile(model.__class__))


@only(["PREDICT"])
def export_outputs(model, features, labels, spec, params):
    probs = spec.predictions["probabilities"]
    classes = tf.constant([model.class_labels])
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
    eval_hooks = spec.evaluation_hooks
    targets = tf.sparse_tensor_to_dense(features["target"], default_value=b"")
    eval_hooks += (
        SaveAttentionWeightVector(
            labels=labels,
            predictions=spec.predictions["class_ids"],
            targets=tf.squeeze(targets, axis=1),
            classes=model.class_labels,
            summary_writer=tf.summary.FileWriterCache.get(
                join(model.run_config.model_dir, "eval")
            ),
            comet=model.comet_experiment,
            n_picks=params.get("n_attn_heatmaps", 5),
            n_hops=params.get("n_hops"),
        ),
    )
    return spec._replace(evaluation_hooks=eval_hooks)


@only(["EVAL"])
def conf_matrix(model, features, labels, spec, params):
    eval_hooks = spec.evaluation_hooks
    eval_metrics = spec.eval_metric_ops or {}
    eval_metrics.update(
        {
            "mean_iou": tf.metrics.mean_iou(
                labels=labels,
                predictions=spec.predictions["class_ids"],
                num_classes=params["n_out_classes"],
            )
        }
    )
    eval_hooks += (
        SaveConfusionMatrix(
            labels=model.class_labels,
            confusion_matrix_tensor_name="mean_iou/total_confusion_matrix",
            summary_writer=tf.summary.FileWriterCache.get(
                join(model.run_config.model_dir, "eval")
            ),
            comet=model.comet_experiment,
        ),
    )
    return spec._replace(
        eval_metric_ops=eval_metrics, evaluation_hooks=eval_hooks
    )


@only(["TRAIN"])
def logging(model, features, labels, spec, params):
    std_metrics = {
        "accuracy": tf.metrics.accuracy(
            labels=labels,
            predictions=spec.predictions["class_ids"],
            name="acc_op",
        ),
        "auc": tf.metrics.auc(
            labels=tf.one_hot(indices=labels, depth=params["n_out_classes"]),
            predictions=spec.predictions["probabilities"],
            name="auc_op",
        ),
    }
    train_hooks = spec.training_hooks or []
    train_hooks += [
        tf.train.LoggingTensorHook(
            tensors={
                "loss": spec.loss,
                "accuracy": std_metrics["accuracy"][1],
                "auc": std_metrics["auc"][1],
            },
            every_n_iter=100,
        )
    ]
    return spec._replace(training_hooks=train_hooks)


@only(["TRAIN"])
def histograms(model, features, labels, spec, params):
    trainable = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    for variable in trainable:
        histogram_name = variable.name.replace(":", "_")
        tf.summary.histogram(histogram_name, variable)
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
            num_classes=params["n_out_classes"],
            name="mpc_acc_op",
        ),
        "auc": tf.metrics.auc(
            labels=tf.one_hot(indices=labels, depth=params["n_out_classes"]),
            predictions=spec.predictions["probabilities"],
            name="auc_op",
        ),
    }
    if spec.mode == ModeKeys.EVAL:
        eval_metrics = spec.eval_metric_ops or {}
        eval_metrics.update(std_metrics)
        spec = spec._replace(eval_metric_ops=eval_metrics)
    tf.summary.scalar("loss", spec.loss)
    tf.summary.scalar("accuracy", std_metrics["accuracy"][1])
    tf.summary.scalar("auc", std_metrics["auc"][1])
    return spec


def early_stopping(model, features, labels, spec, params):
    makedirs(model.estimator.eval_dir())
    train_hooks = spec.training_hooks or []
    train_hooks += [
        stop_if_no_decrease_hook(
            estimator=model.estimator,
            metric_name="loss",
            max_steps_without_decrease=params.get("max_steps", 1000),
            min_steps=params.get("min_steps", 100),
        )
    ]
    return spec._replace(training_hooks=train_hooks)