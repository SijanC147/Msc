import tensorflow as tf
from os.path import join, dirname, exists
from inspect import getsource, getfile
from datetime import timedelta
from time import time as _time
from abc import ABC, abstractmethod


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
    ):
        self.params = params
        self.feature_columns = feature_columns
        self.train_input_fn = train_input_fn
        self.train_hooks = train_hooks
        self.eval_input_fn = eval_input_fn
        self.eval_hooks = eval_hooks
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
    def estimator(self):
        self.__estimator = tf.estimator.Estimator(
            model_fn=self.model_fn,
            params={"feature_columns": self.feature_columns, **self.params},
            config=self.run_config,
        )
        return self.__estimator

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
        self.__model_fn = model_fn

    @train_hooks.setter
    def train_hooks(self, train_hooks=None):
        self.__train_hooks = train_hooks

    @eval_hooks.setter
    def eval_hooks(self, eval_hooks=None):
        self.__eval_hooks = eval_hooks

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
    def _model_fn(self):
        pass

    def _train_hooks(self):
        return []

    def _eval_hooks(self):
        return []

    def train(self, dataset, steps, distribution=None, hooks=[]):
        self._add_embedding_params(embedding=dataset.embedding)
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
            hooks=self.train_hooks + hooks,
        )
        time_taken = str(timedelta(seconds=_time() - start))
        duration_dict = {"job": "train", "time": time_taken}
        return {"duration": duration_dict, **run_stats}

    def evaluate(self, dataset, distribution=None, hooks=[]):
        self._add_embedding_params(embedding=dataset.embedding)
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
            hooks=self.eval_hooks + hooks,
        )
        time_taken = str(timedelta(seconds=_time() - start))
        duration_dict = {"job": "eval", "time": time_taken}
        return {"duration": duration_dict, **run_stats}

    def train_and_eval(self, dataset, steps):
        self._add_embedding_params(embedding=dataset.embedding)
        features, labels, stats = dataset.get_features_and_labels(mode="train")
        train_stats = self._export_statistics(dataset_stats=stats, steps=steps)
        train_spec = tf.estimator.TrainSpec(
            input_fn=lambda: self.__train_input_fn(
                features=features,
                labels=labels,
                batch_size=self.params["batch_size"],
            ),
            max_steps=steps,
            hooks=self.train_hooks,
        )
        features, labels, stats = dataset.get_features_and_labels(mode="eval")
        eval_stats = self._export_statistics(dataset_stats=stats, steps=steps)
        eval_spec = tf.estimator.EvalSpec(
            input_fn=lambda: self.__eval_input_fn(
                features=features,
                labels=labels,
                batch_size=self.params["batch_size"],
            ),
            steps=None,
            hooks=self.eval_hooks,
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
            "embedding_initializer": embedding.initializer,
            "vocab_size": embedding.vocab_size,
            "embedding_dim": embedding.dim_size,
            **self.params,
        }
