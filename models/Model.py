import os
import tensorflow as tf
import inspect
import time
import datetime
from abc import ABC, abstractmethod


class Model(ABC):
    def __init__(self, embedding=None, dataset=None, run_config=None):
        self.embedding = embedding
        self.dataset = dataset
        self.run_config = run_config
        self.estimator = None

    @abstractmethod
    def set_params(self, params):
        self.params = {
            "embedding_initializer": self.embedding.initializer,
            "vocab_size": self.embedding.vocab_size,
            "embedding_dim": self.embedding.dim_size,
            **params,
        }

    @abstractmethod
    def set_feature_columns(self, feature_columns):
        self.feature_columns = feature_columns

    @abstractmethod
    def set_train_input_fn(self, train_input_fn):
        self.train_input_fn = train_input_fn

    @abstractmethod
    def set_eval_input_fn(self, eval_input_fn):
        self.eval_input_fn = eval_input_fn

    @abstractmethod
    def set_model_fn(self, model_fn):
        self.model_fn = model_fn

    def initialize_internal_defaults(self):
        self.set_feature_columns(feature_columns=None)
        self.set_params(params=None)
        self.set_train_input_fn(train_input_fn=None)
        self.set_eval_input_fn(eval_input_fn=None)
        self.set_model_fn(model_fn=None)

    def create_estimator(self):
        self.estimator = tf.estimator.Estimator(
            model_fn=self.model_fn,
            params={"feature_columns": self.feature_columns, **self.params},
            config=self.run_config,
        )

    def train(self, steps, hooks=None, debug=False, distribution=None):
        mode = "train" if not debug else "debug"
        features, labels, stats = self.dataset.get_features_and_labels(
            mode=mode, distribution=distribution
        )
        run_stats = self.export_statistics(
            dataset_stats=stats, steps=steps, train_hooks=hooks
        )
        start = time.time()
        self.estimator.train(
            input_fn=lambda: self.train_input_fn(
                features=features,
                labels=labels,
                batch_size=self.params["batch_size"],
            ),
            steps=steps,
            hooks=hooks,
        )
        time_taken = str(datetime.timedelta(seconds=time.time() - start))
        duration_dict = {"job": "train", "time": time_taken}
        return {"duration": duration_dict, **run_stats}

    def evaluate(self, hooks=None, debug=False, distribution=None):
        mode = "eval" if not debug else "debug"
        features, labels, stats = self.dataset.get_features_and_labels(
            mode=mode, distribution=distribution
        )
        run_stats = self.export_statistics(
            dataset_stats=stats, eval_hooks=hooks
        )
        start = time.time()
        self.estimator.evaluate(
            input_fn=lambda: self.eval_input_fn(
                features=features, labels=labels
            ),
            hooks=hooks,
        )
        time_taken = str(datetime.timedelta(seconds=time.time() - start))
        duration_dict = {"job": "eval", "time": time_taken}
        return {"duration": duration_dict, **run_stats}

    def train_and_evaluate(
        self,
        steps=None,
        train_hooks=None,
        eval_hooks=None,
        train_distribution=None,
        eval_distribution=None,
    ):
        features, labels, stats = self.dataset.get_features_and_labels(
            mode="train", distribution=train_distribution
        )
        train_stats = self.export_statistics(
            dataset_stats=stats,
            steps=steps,
            train_hooks=train_hooks,
            eval_hooks=eval_hooks,
        )
        train_spec = tf.estimator.TrainSpec(
            input_fn=lambda: self.train_input_fn(
                features=features,
                labels=labels,
                batch_size=self.params["batch_size"],
            ),
            max_steps=steps,
            hooks=train_hooks,
        )
        features, labels, stats = self.dataset.get_features_and_labels(
            mode="eval", distribution=eval_distribution
        )
        eval_stats = self.export_statistics(
            dataset_stats=stats,
            steps=steps,
            train_hooks=train_hooks,
            eval_hooks=eval_hooks,
        )
        eval_spec = tf.estimator.EvalSpec(
            input_fn=lambda: self.eval_input_fn(
                features=features, labels=labels
            ),
            steps=None,
            hooks=eval_hooks,
        )
        start = time.time()
        tf.estimator.train_and_evaluate(
            estimator=self.estimator,
            train_spec=train_spec,
            eval_spec=eval_spec,
        )
        time_taken = str(datetime.timedelta(seconds=time.time() - start))
        duration_dict = {"job": "train+eval", "time": time_taken}
        return (
            {"duration": duration_dict, **train_stats},
            {"duration": duration_dict, **eval_stats},
        )

    def export_statistics(
        self, dataset_stats=None, steps=None, train_hooks=None, eval_hooks=None
    ):
        train_input_fn_source = inspect.getsource(self.train_input_fn)
        eval_input_fn_source = inspect.getsource(self.eval_input_fn)
        model_fn_source = inspect.getsource(self.model_fn)
        model_common_file = os.path.join(
            os.path.dirname(inspect.getfile(self.__class__)), "common.py"
        )
        estimator_train_fn_source = inspect.getsource(self.train)
        estimator_eval_fn_source = inspect.getsource(self.evaluate)
        estimator_train_eval_fn_source = inspect.getsource(
            self.train_and_evaluate
        )
        if os.path.exists(model_common_file):
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
                "train_hooks": train_hooks,
                "eval_hooks": eval_hooks,
                "train_fn": estimator_train_fn_source,
                "eval_fn": estimator_eval_fn_source,
                "train_eval_fn": estimator_train_eval_fn_source,
            },
            "common": common_content,
        }
