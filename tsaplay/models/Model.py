import tensorflow as tf
from tensorflow.estimator import ModeKeys  # pylint: disable=E0401
from tensorflow.saved_model.signature_constants import (
    DEFAULT_SERVING_SIGNATURE_DEF_KEY
)
from tensorflow.estimator.export import (  # pylint: disable=E0401
    PredictOutput,
    RegressionOutput,
    ClassificationOutput,
)
from tensorflow.contrib.estimator import (  # pylint: disable=E0611
    stop_if_no_decrease_hook
)
from os.path import join, dirname, exists
from inspect import getsource, getfile
from datetime import timedelta
from time import time as _time
from abc import ABC, abstractmethod
from os import makedirs
from functools import wraps
from tsaplay.utils._tf import get_dense_tensor
from tsaplay.utils.SaveConfusionMatrixHook import SaveConfusionMatrixHook
from tsaplay.utils.SaveAttentionWeightVectorHook import (
    SaveAttentionWeightVectorHook
)


class Model(ABC):
    def __init__(
        self,
        run_config=None,
        params=None,
        feature_columns=None,
        train_input_fn=None,
        train_hooks=None,
        eval_input_fn=None,
        eval_hooks=None,
        model_fn=None,
        serving_input_fn=None,
    ):
        self.params = params
        self.feature_columns = feature_columns
        self.train_input_fn = train_input_fn
        self.train_hooks = train_hooks
        self.eval_input_fn = eval_input_fn
        self.eval_hooks = eval_hooks
        self.serving_input_fn = serving_input_fn
        self.model_fn = model_fn
        self.run_config = run_config

    @property
    def params(self):
        if self.__params is None:
            self.params = self._params()
        return self.__params

    @property
    def feature_columns(self):
        if self.__feature_columns is None:
            self.feature_columns = self._feature_columns()
        return self.__feature_columns

    @property
    def train_input_fn(self):
        if self.__train_input_fn is None:
            self.train_input_fn = self._train_input_fn()
        return self.__train_input_fn

    @property
    def eval_input_fn(self):
        if self.__eval_input_fn is None:
            self.eval_input_fn = self._eval_input_fn()
        return self.__eval_input_fn

    @property
    def model_fn(self):
        if self.__model_fn is None:
            self.model_fn = self._model_fn()
        return self.__model_fn

    @property
    def train_hooks(self):
        if self.__train_hooks is None:
            self.train_hooks = self._train_hooks()
        return self.__train_hooks

    @property
    def eval_hooks(self):
        if self.__eval_hooks is None:
            self.eval_hooks = self._eval_hooks()
        return self.__eval_hooks

    @property
    def serving_input_fn(self):
        if self.__serving_input_fn is None:
            self.serving_input_fn = self._serving_input_fn()
        return self.__serving_input_fn

    @property
    def estimator(self):
        self.__estimator = tf.estimator.Estimator(
            model_fn=self.model_fn,
            params={"feature_columns": self.feature_columns, **self.params},
            config=self.run_config,
        )
        return self.__estimator

    @property
    def run_config(self):
        return self.__run_config

    @params.setter
    def params(self, params):
        self.__params = params

    @feature_columns.setter
    def feature_columns(self, feature_columns):
        self.__feature_columns = feature_columns

    @train_input_fn.setter
    def train_input_fn(self, train_input_fn):
        self.__train_input_fn = train_input_fn

    @eval_input_fn.setter
    def eval_input_fn(self, eval_input_fn):
        self.__eval_input_fn = eval_input_fn

    @model_fn.setter
    def model_fn(self, model_fn):
        if model_fn is not None:
            self.__model_fn = self._wrap_model_fn(model_fn)
        else:
            self.__model_fn = None

    @train_hooks.setter
    def train_hooks(self, train_hooks):
        if train_hooks is None:
            train_hooks = []
        self.__train_hooks = train_hooks

    @eval_hooks.setter
    def eval_hooks(self, eval_hooks):
        if eval_hooks is None:
            eval_hooks = []
        self.__eval_hooks = eval_hooks

    @serving_input_fn.setter
    def serving_input_fn(self, serving_input_fn):
        self.__serving_input_fn = serving_input_fn

    @run_config.setter
    def run_config(self, run_config):
        if run_config is None:
            self.__run_config = tf.estimator.RunConfig()
        else:
            self.__run_config = run_config

    @abstractmethod
    def _params(self):
        pass

    @abstractmethod
    def _feature_columns(self):
        pass

    @abstractmethod
    def _train_input_fn(self):
        pass

    @abstractmethod
    def _eval_input_fn(self):
        pass

    @abstractmethod
    def _serving_input_fn(self):
        pass

    @abstractmethod
    def _model_fn(self):
        pass

    def _train_hooks(self):
        return []

    def _eval_hooks(self):
        return []

    def train(self, dataset, embedding, steps, distribution=None, hooks=[]):
        self._add_embedding_params(embedding)
        if "partial" in embedding.name and not (
            exists(join(embedding.data_dir, "train.tfrecord"))
        ):
            sentence_list = tf.train.BytesList(
                value=[s.encode() for s in dataset.train_dict["sentences"]]
            )
            target_list = tf.train.BytesList(
                value=[t.encode() for t in dataset.train_dict["targets"]]
            )
            label_list = tf.train.Int64List(
                value=[int(l) for l in dataset.train_dict["labels"]]
            )
            sentences = tf.train.Feature(bytes_list=sentence_list)
            targets = tf.train.Feature(bytes_list=target_list)
            labels = tf.train.Feature(int64_list=label_list)

            train_dict = {
                "sentences": sentences,
                "targets": targets,
                "labels": labels,
            }

            dataset = tf.train.Features(feature=train_dict)
            example = tf.train.Example(features=dataset)

            with tf.python_io.TFRecordWriter("sentneces.tfrecord") as writer:
                writer.write(example.SerializeToString())

        features, labels, stats = dataset.get_features_and_labels(
            mode="train", distribution=distribution
        )
        run_stats = self._export_statistics(dataset_stats=stats, steps=steps)
        start = _time()
        self.estimator.train(
            input_fn=lambda: self.__train_input_fn(
                features=features,
                labels=labels,
                batch_size=self.params["batch_size"],
            ),
            steps=steps,
            hooks=self._attach_std_train_hooks(self.train_hooks) + hooks,
        )
        time_taken = str(timedelta(seconds=_time() - start))
        duration_dict = {"job": "train", "time": time_taken}
        return {"duration": duration_dict, **run_stats}

    def evaluate(self, dataset, embedding, distribution=None, hooks=[]):
        self._add_embedding_params(embedding)
        features, labels, stats = dataset.get_features_and_labels(
            mode="test", distribution=distribution
        )
        run_stats = self._export_statistics(dataset_stats=stats)
        start = _time()
        self.estimator.evaluate(
            input_fn=lambda: self.__eval_input_fn(
                features=features,
                labels=labels,
                batch_size=self.params["batch_size"],
            ),
            hooks=self._attach_std_eval_hooks(self.eval_hooks) + hooks,
        )
        time_taken = str(timedelta(seconds=_time() - start))
        duration_dict = {"job": "eval", "time": time_taken}
        return {"duration": duration_dict, **run_stats}

    def train_and_eval(self, dataset, embedding, steps, early_stopping=False):
        self._add_embedding_params(embedding)
        features, labels, stats = dataset.get_features_and_labels(mode="train")
        features["vocab_file"] = dataset.embedding.vocab_file_path
        train_stats = self._export_statistics(dataset_stats=stats, steps=steps)
        train_spec = tf.estimator.TrainSpec(
            input_fn=lambda: self.__train_input_fn(
                features=features,
                labels=labels,
                batch_size=self.params["batch_size"],
            ),
            max_steps=steps,
            hooks=self._attach_std_train_hooks(self.train_hooks)
            + self._get_early_stopping_hook(early_stopping),
        )
        features, labels, stats = dataset.get_features_and_labels(mode="eval")
        features["vocab_file"] = dataset.embedding.vocab_file_path
        eval_stats = self._export_statistics(dataset_stats=stats, steps=steps)
        eval_spec = tf.estimator.EvalSpec(
            input_fn=lambda: self.__eval_input_fn(
                features=features,
                labels=labels,
                batch_size=self.params["batch_size"],
            ),
            steps=None,
            hooks=self._attach_std_eval_hooks(self.eval_hooks),
        )
        start = _time()
        tf.estimator.train_and_evaluate(
            estimator=self.estimator,
            train_spec=train_spec,
            eval_spec=eval_spec,
        )
        time_taken = str(timedelta(seconds=_time() - start))
        duration_dict = {"job": "train+eval", "time": time_taken}
        return (
            {"duration": duration_dict, **train_stats},
            {"duration": duration_dict, **eval_stats},
        )

    def export(self, directory, embedding):
        self._add_embedding_params(embedding)
        self.estimator.export_savedmodel(
            export_dir_base=directory,
            serving_input_receiver_fn=self._serving_input_receiver_fn(),
            assets_extra={"vocab_file": embedding.vocab_file_path},
            strip_default_attrs=True,
        )

    def _serving_input_receiver_fn(self):
        self.serving_input_fn = self._serving_input_fn()

        def serving_input_receiver_fn():
            inputs_serialized = tf.placeholder(dtype=tf.string)

            feature_spec = {
                "sen_lit": tf.FixedLenFeature(dtype=tf.string, shape=[]),
                "target_lit": tf.FixedLenFeature(dtype=tf.string, shape=[]),
                "left_lit": tf.FixedLenFeature(dtype=tf.string, shape=[]),
                "right_lit": tf.FixedLenFeature(dtype=tf.string, shape=[]),
                "sen_tok": tf.FixedLenFeature(dtype=tf.string, shape=[]),
                "target_tok": tf.FixedLenFeature(dtype=tf.string, shape=[]),
                "left_tok": tf.FixedLenFeature(dtype=tf.string, shape=[]),
                "right_tok": tf.FixedLenFeature(dtype=tf.string, shape=[]),
                "sen_len": tf.FixedLenFeature(dtype=tf.int64, shape=[]),
                "left_len": tf.FixedLenFeature(dtype=tf.int64, shape=[]),
                "right_len": tf.FixedLenFeature(dtype=tf.int64, shape=[]),
                "target_len": tf.FixedLenFeature(dtype=tf.int64, shape=[]),
                "left_map": tf.VarLenFeature(dtype=tf.int64),
                "right_map": tf.VarLenFeature(dtype=tf.int64),
                "target_map": tf.VarLenFeature(dtype=tf.int64),
                "sen_map": tf.VarLenFeature(dtype=tf.int64),
                "ctxt_map": tf.VarLenFeature(dtype=tf.int64),
                "lft_trg_map": tf.VarLenFeature(dtype=tf.int64),
                "trg_rht_map": tf.VarLenFeature(dtype=tf.int64),
            }

            input_features = tf.parse_example(inputs_serialized, feature_spec)

            left_map = get_dense_tensor(input_features["left_map"])
            target_map = get_dense_tensor(input_features["target_map"])
            right_map = get_dense_tensor(input_features["right_map"])
            sen_map = get_dense_tensor(input_features["sen_map"])
            ctxt_map = get_dense_tensor(input_features["ctxt_map"])
            lft_trg_map = get_dense_tensor(input_features["lft_trg_map"])
            trg_rht_map = get_dense_tensor(input_features["trg_rht_map"])

            std_feat = {
                "literals": {
                    "sentence": input_features["sen_lit"],
                    "target": input_features["target_lit"],
                    "left": input_features["left_lit"],
                    "right": input_features["right_lit"],
                },
                "tok_enc": {
                    "sentence": input_features["sen_tok"],
                    "target": input_features["target_tok"],
                    "left": input_features["left_tok"],
                    "right": input_features["right_tok"],
                },
                "lengths": {
                    "sentence": tf.cast(input_features["sen_len"], tf.int32),
                    "left": tf.cast(input_features["left_len"], tf.int32),
                    "right": tf.cast(input_features["right_len"], tf.int32),
                    "target": tf.cast(input_features["target_len"], tf.int32),
                },
                "mappings": {
                    "left": left_map,
                    "target": target_map,
                    "right": right_map,
                    "sentence": sen_map,
                    "context": ctxt_map,
                    "left_target": lft_trg_map,
                    "target_right": trg_rht_map,
                },
            }

            input_feat = self.__serving_input_fn(std_feat)

            inputs = {"instances": inputs_serialized}

            return tf.estimator.export.ServingInputReceiver(input_feat, inputs)

        return serving_input_receiver_fn

    def _wrap_model_fn(self, _model_fn):
        @wraps(_model_fn)
        def wrapper(features, labels, mode, params):
            if mode == ModeKeys.PREDICT:
                params["keep_prob"] = 1
            spec = _model_fn(features, labels, mode, params)
            if mode == ModeKeys.PREDICT:
                probs = spec.predictions["probabilities"]
                classes = tf.constant([["Negative", "Neutral", "Positive"]])
                classify_output = ClassificationOutput(
                    classes=classes, scores=probs
                )
                predict_output = PredictOutput(
                    {**spec.predictions, **features}
                )
                export_outputs = {
                    DEFAULT_SERVING_SIGNATURE_DEF_KEY: classify_output,
                    "inspect": predict_output,
                }
                all_export_outputs = spec.export_outputs or {}
                all_export_outputs.update(export_outputs)
                return spec._replace(export_outputs=all_export_outputs)
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
                    labels=tf.one_hot(
                        indices=labels, depth=params["n_out_classes"]
                    ),
                    predictions=spec.predictions["probabilities"],
                    name="auc_op",
                ),
                "mean_iou": tf.metrics.mean_iou(
                    labels=labels,
                    predictions=spec.predictions["class_ids"],
                    num_classes=params["n_out_classes"],
                ),
            }
            tf.summary.scalar("accuracy", std_metrics["accuracy"][1])
            tf.summary.scalar("auc", std_metrics["auc"][1])
            if mode == ModeKeys.EVAL:
                all_eval_hooks = spec.evaluation_hooks or []
                if features.get("target_lit") is not None:
                    attn_hook = SaveAttentionWeightVectorHook(
                        labels=labels,
                        predictions=spec.predictions["class_ids"],
                        targets=features["target_lit"],
                        summary_writer=tf.summary.FileWriterCache.get(
                            join(self.run_config.model_dir, "eval")
                        ),
                        n_picks=self.params.get("n_attn_heatmaps", 5),
                        n_hops=self.params.get("n_hops"),
                    )
                    all_eval_hooks += [attn_hook]
                all_metrics = spec.eval_metric_ops or {}
                all_metrics.update(std_metrics)
                return spec._replace(
                    eval_metric_ops=all_metrics,
                    evaluation_hooks=all_eval_hooks,
                )
            if mode == ModeKeys.TRAIN:
                tf.summary.scalar("loss", spec.loss)
                trainable = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
                for variable in trainable:
                    histogram_name = variable.name.replace(":", "_")
                    tf.summary.histogram(histogram_name, variable)
                logging_hook = tf.train.LoggingTensorHook(
                    tensors={
                        "loss": spec.loss,
                        "accuracy": std_metrics["accuracy"][1],
                        "auc": std_metrics["auc"][1],
                    },
                    every_n_iter=100,
                )
                all_training_hooks = spec.training_hooks or []
                all_training_hooks += [logging_hook]
                return spec._replace(training_hooks=all_training_hooks)

        return wrapper

    def _export_statistics(self, dataset_stats=None, steps=None):
        train_input_fn_source = getsource(self.train_input_fn)
        eval_input_fn_source = getsource(self.eval_input_fn)
        model_fn_source = getsource(self.model_fn)
        model_common_file = join(dirname(getfile(self.__class__)), "common.py")
        estimator_train_fn_source = getsource(self.train)
        estimator_eval_fn_source = getsource(self.evaluate)
        estimator_train_eval_fn_source = getsource(self.train_and_eval)
        if exists(model_common_file):
            common_content = open(model_common_file, "r").read()
        else:
            common_content = ""
        return {
            "dataset": dataset_stats,
            "steps": steps,
            "model": {
                "params": self.params,
                "train_input_fn": train_input_fn_source,
                "eval_input_fn": eval_input_fn_source,
                "model_fn": model_fn_source,
            },
            "estimator": {
                "train_hooks": self.train_hooks,
                "eval_hooks": self.eval_hooks,
                "train_fn": estimator_train_fn_source,
                "eval_fn": estimator_eval_fn_source,
                "train_eval_fn": estimator_train_eval_fn_source,
            },
            "common": common_content,
        }

    def _add_embedding_params(self, embedding):
        self.params = {
            **self.params,
            "embedding_initializer": embedding.initializer,
            "vocab_size": embedding.vocab_size,
            "embedding_dim": embedding.dim_size,
            "vocab_file_path": embedding.vocab_file_path,
        }

    def _attach_std_eval_hooks(self, eval_hooks):
        confusion_matrix_save_hook = SaveConfusionMatrixHook(
            labels=["Negative", "Neutral", "Positive"],
            confusion_matrix_tensor_name="mean_iou/total_confusion_matrix",
            summary_writer=tf.summary.FileWriterCache.get(
                join(self.run_config.model_dir, "eval")
            ),
        )
        std_eval_hooks = [confusion_matrix_save_hook]
        return eval_hooks + std_eval_hooks

    def _attach_std_train_hooks(self, train_hooks):
        std_train_hooks = []
        return train_hooks + std_train_hooks

    def _get_early_stopping_hook(self, early_stopping):
        if early_stopping or self.params.get("early_stopping", False):
            makedirs(self.estimator.eval_dir())
            early_stopping_hook = [
                stop_if_no_decrease_hook(
                    estimator=self.estimator,
                    metric_name="loss",
                    max_steps_without_decrease=self.params.get(
                        "max_steps", 1000
                    ),
                    min_steps=self.params.get("min_steps", 100),
                )
            ]
        else:
            early_stopping_hook = []

        return early_stopping_hook
