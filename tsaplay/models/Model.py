import tensorflow as tf
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
from tensorflow.contrib.estimator import (  # pylint: disable=E0611
    stop_if_no_decrease_hook
)
from os.path import join, dirname, exists, relpath
from inspect import getsource, getfile
from abc import ABC, abstractmethod
from os import makedirs, getcwd
from functools import wraps
from tsaplay.models._decorators import attach_embedding_params
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
    def name(self):
        return self.__class__.__name__

    @property
    def relative_path(self):
        models_dir = join(getcwd(), "tsaplay", "models")
        model_dir = dirname(getfile(self.__class__))
        relative_path = relpath(model_dir, models_dir)
        return join(relative_path, self.name)

    @property
    def params(self):
        return {**self.__params, "feature_columns": self.feature_columns}

    @property
    def feature_columns(self):
        return self.__feat_col

    @property
    def train_input_fn(self):
        return self.__train_in_fn

    @property
    def eval_input_fn(self):
        return self.__eval_in_fn

    @property
    def model_fn(self):
        return self.__model_fn

    @property
    def train_hooks(self):
        return self.__train_hooks

    @property
    def eval_hooks(self):
        return self.__eval_hooks

    @property
    def serving_input_fn(self):
        return self.__serv_in_fn

    @property
    def run_config(self):
        return self.__run_conf

    @property
    def estimator(self):
        return Estimator(
            model_fn=self.model_fn, params=self.params, config=self.run_config
        )

    @params.setter
    def params(self, params):
        self.__params = params or self._params()

    @feature_columns.setter
    def feature_columns(self, feature_columns):
        self.__feat_col = feature_columns or self._feature_columns()

    @train_input_fn.setter
    def train_input_fn(self, train_input_fn):
        self.__train_in_fn = train_input_fn or self._train_input_fn()

    @eval_input_fn.setter
    def eval_input_fn(self, eval_input_fn):
        self.__eval_in_fn = eval_input_fn or self._eval_input_fn()

    @model_fn.setter
    def model_fn(self, model_fn):
        self.__model_fn = self._wrap_model_fn(model_fn or self._model_fn())

    @train_hooks.setter
    def train_hooks(self, train_hooks):
        self.__train_hooks = train_hooks or self._train_hooks()

    @eval_hooks.setter
    def eval_hooks(self, eval_hooks):
        self.__eval_hooks = eval_hooks or self._eval_hooks()

    @serving_input_fn.setter
    def serving_input_fn(self, serving_input_fn):
        self.__serv_in_fn = serving_input_fn or self._serving_input_fn()

    @run_config.setter
    def run_config(self, run_config):
        self.__run_conf = run_config or self._run_config()

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

    def _run_config(self):
        return RunConfig()

    @attach_embedding_params
    def train(self, feature_provider, steps):
        self.estimator.train(
            input_fn=lambda: self.__train_in_fn(
                tfrecord=feature_provider.get_tfrecord("train"),
                batch_size=self.params["batch_size"],
            ),
            steps=steps,
            hooks=self._attach_std_train_hooks(self.train_hooks),
        )

    @attach_embedding_params
    def evaluate(self, feature_provider):
        self.estimator.evaluate(
            input_fn=lambda: self.__eval_in_fn(
                tfrecord=feature_provider.get_tfrecord("test"),
                batch_size=self.params["batch_size"],
            ),
            hooks=self._attach_std_eval_hooks(self.eval_hooks),
        )

    @attach_embedding_params
    def train_and_eval(self, feature_provider, steps):
        stop_early = self.params.get("early_stopping", False)
        train_spec = tf.estimator.TrainSpec(
            input_fn=lambda: self.__train_in_fn(
                tfrecord=feature_provider.get_tfrecord("train"),
                batch_size=self.params["batch_size"],
            ),
            max_steps=steps,
            hooks=self._attach_std_train_hooks(self.train_hooks, stop_early),
        )
        eval_spec = tf.estimator.EvalSpec(
            input_fn=lambda: self.__eval_in_fn(
                tfrecord=feature_provider.get_tfrecord("test"),
                batch_size=self.params["batch_size"],
            ),
            steps=None,
            hooks=self._attach_std_eval_hooks(self.eval_hooks),
        )
        tf.estimator.train_and_evaluate(
            estimator=self.estimator,
            train_spec=train_spec,
            eval_spec=eval_spec,
        )

    @attach_embedding_params
    def export(self, directory, embedding_params):
        self.estimator.export_savedmodel(
            export_dir_base=directory,
            serving_input_receiver_fn=self._serving_input_receiver_fn(),
            strip_default_attrs=True,
        )

    def _serving_input_receiver_fn(self):
        # self.serving_input_fn = self._serving_input_fn()

        def serving_input_receiver_fn():
            inputs_serialized = tf.placeholder(dtype=tf.string)

            feature_spec = {
                "left": tf.VarLenFeature(dtype=tf.string),
                "target": tf.VarLenFeature(dtype=tf.string),
                "right": tf.VarLenFeature(dtype=tf.string),
                "left_ids": tf.VarLenFeature(dtype=tf.int64),
                "target_ids": tf.VarLenFeature(dtype=tf.int64),
                "right_ids": tf.VarLenFeature(dtype=tf.int64),
            }

            parsed_example = tf.parse_example(inputs_serialized, feature_spec)

            features = {
                "left": parsed_example["left"],
                "target": parsed_example["target"],
                "right": parsed_example["right"],
                "left_ids": parsed_example["left_ids"],
                "target_ids": parsed_example["target_ids"],
                "right_ids": parsed_example["right_ids"],
            }

            # left_map = tf.sparse_tensor_to_dense(input_features["left_map"])
            # target_map = tf.sparse_tensor_to_dense(
            #     input_features["target_map"]
            # )
            # right_map = tf.sparse_tensor_to_dense(input_features["right_map"])
            # sen_map = tf.sparse_tensor_to_dense(input_features["sen_map"])
            # ctxt_map = tf.sparse_tensor_to_dense(input_features["ctxt_map"])
            # lft_trg_map = tf.sparse_tensor_to_dense(
            #     input_features["lft_trg_map"]
            # )
            # trg_rht_map = tf.sparse_tensor_to_dense(
            #     input_features["trg_rht_map"]
            # )

            # std_feat = {
            #     "literals": {
            #         "sentence": input_features["sen_lit"],
            #         "target": input_features["target_lit"],
            #         "left": input_features["left_lit"],
            #         "right": input_features["right_lit"],
            #     },
            #     "tok_enc": {
            #         "sentence": input_features["sen_tok"],
            #         "target": input_features["target_tok"],
            #         "left": input_features["left_tok"],
            #         "right": input_features["right_tok"],
            #     },
            #     "lengths": {
            #         "sentence": tf.cast(input_features["sen_len"], tf.int32),
            #         "left": tf.cast(input_features["left_len"], tf.int32),
            #         "right": tf.cast(input_features["right_len"], tf.int32),
            #         "target": tf.cast(input_features["target_len"], tf.int32),
            #     },
            #     "mappings": {
            #         "left": left_map,
            #         "target": target_map,
            #         "right": right_map,
            #         "sentence": sen_map,
            #         "context": ctxt_map,
            #         "left_target": lft_trg_map,
            #         "target_right": trg_rht_map,
            #     },
            # }

            input_feat = self.__serv_in_fn(features)

            inputs = {"instances": inputs_serialized}

            return ServingInputReceiver(input_feat, inputs)

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

    def _attach_std_train_hooks(self, train_hooks, early_stopping=False):
        std_train_hooks = []
        if early_stopping:
            makedirs(self.estimator.eval_dir())
            std_train_hooks.append(
                [
                    stop_if_no_decrease_hook(
                        estimator=self.estimator,
                        metric_name="loss",
                        max_steps_without_decrease=self.params.get(
                            "max_steps", 1000
                        ),
                        min_steps=self.params.get("min_steps", 100),
                    )
                ]
            )

        return train_hooks + std_train_hooks
