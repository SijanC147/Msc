import os
import tensorflow as tf
import inspect
import time
import datetime
from abc import ABC, abstractmethod


class Model(ABC):
    def __init__(self, embedding=None, dataset=None, model_dir=None):
        self.embedding = embedding
        self.dataset = dataset
        self.model_dir = model_dir
        self.estimator = None

    @abstractmethod
    def set_params(self, params):
        self.params = {
            "feature_columns": self.feature_columns,
            "embedding_initializer": self.embedding.get_tf_embedding_initializer(),
            "vocab_size": self.embedding.vocab_size,
            "embedding_dim": self.embedding.dimension_size,
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

    def train(
        self,
        steps,
        batch_size=None,
        hooks=None,
        debug=False,
        label_distribution=None,
    ):
        batch = (
            batch_size if batch_size is not None else self.params["batch_size"]
        )
        if not (debug):
            features, labels, stats = self.dataset.get_mapped_features_and_labels(
                mode="train", distribution=label_distribution
            )
        else:
            features, labels, stats = self.dataset.get_mapped_features_and_labels(
                mode="debug", distribution=label_distribution
            )
        self.init_estimator_if_none()
        run_stats = self.export_statistics(
            dataset_stats=stats,
            batch_size=batch_size,
            steps=steps,
            train_hooks=hooks,
        )
        print("Training {0}...".format(self.__class__.__name__))
        start = time.time()
        self.estimator.train(
            input_fn=lambda: self.train_input_fn(
                features=features, labels=labels, batch_size=batch
            ),
            steps=steps,
            hooks=hooks,
        )
        time_taken = str(datetime.timedelta(seconds=time.time() - start))
        duration_dict = {"job": "train", "time": time_taken}
        print(
            "Finished training {0} in {1}".format(
                self.__class__.__name__, time_taken
            )
        )
        return {"duration": duration_dict, **run_stats}

    def evaluate(self, hooks=None, debug=False, label_distribution=None):
        if not (debug):
            features, labels, stats = self.dataset.get_mapped_features_and_labels(
                mode="eval", distribution=label_distribution
            )
        else:
            features, labels, stats = self.dataset.get_mapped_features_and_labels(
                mode="debug", distribution=label_distribution
            )
        self.init_estimator_if_none()
        run_stats = self.export_statistics(
            dataset_stats=stats, eval_hooks=hooks
        )
        print("Evaluating {0}...".format(self.__class__.__name__))
        start = time.time()
        self.estimator.evaluate(
            input_fn=lambda: self.eval_input_fn(
                features=features, labels=labels
            ),
            hooks=hooks,
        )
        time_taken = str(datetime.timedelta(seconds=time.time() - start))
        print(
            "Finished evaluating {0} in {1}".format(
                self.__class__.__name__, time_taken
            )
        )
        duration_dict = {"job": "eval", "time": time_taken}
        return {"duration": duration_dict, **run_stats}

    def train_and_evaluate(
        self,
        steps=None,
        batch_size=None,
        train_hooks=None,
        eval_hooks=None,
        train_distribution=None,
        eval_distribution=None,
    ):

        if batch_size is not None and batch_size != self.params["batch_size"]:
            self.params["batch_size"] = batch_size

        train_features, train_labels, train_stats = self.dataset.get_mapped_features_and_labels(
            mode="train", distribution=train_distribution
        )
        eval_features, eval_labels, eval_stats = self.dataset.get_mapped_features_and_labels(
            mode="eval", distribution=eval_distribution
        )
        self.init_estimator_if_none()
        train_run_stats = self.export_statistics(
            dataset_stats=train_stats,
            batch_size=batch_size,
            steps=steps,
            train_hooks=train_hooks,
            eval_hooks=eval_hooks,
        )
        eval_run_stats = self.export_statistics(
            dataset_stats=eval_stats,
            batch_size=batch_size,
            steps=steps,
            train_hooks=train_hooks,
            eval_hooks=eval_hooks,
        )
        # os.makedirs(self.estimator.eval_dir(), exist_ok=True)
        # early_stopping = tf.contrib.estimator.stop_if_no_decrease_hook(
        #     self.estimator,
        #     metric_name='loss',
        #     max_steps_without_decrease=10,
        #     min_steps=300)
        print(
            "{0} starting to train and evaluate...".format(
                self.__class__.__name__
            )
        )
        start = time.time()
        tf.estimator.train_and_evaluate(
            estimator=self.estimator,
            train_spec=tf.estimator.TrainSpec(
                input_fn=lambda: self.train_input_fn(
                    features=train_features,
                    labels=train_labels,
                    batch_size=self.params["batch_size"],
                ),
                max_steps=steps,
                # hooks=[early_stopping] if train_hooks==None else [early_stopping]+train_hooks
                hooks=train_hooks,
            ),
            eval_spec=tf.estimator.EvalSpec(
                input_fn=lambda: self.eval_input_fn(
                    features=eval_features, labels=eval_labels
                ),
                steps=None,
                hooks=eval_hooks,
            ),
        )
        time_taken = str(datetime.timedelta(seconds=time.time() - start))
        print(
            "{0} trained and evaluated in {1}".format(
                self.__class__.__name__, time_taken
            )
        )
        duration_dict = {"job": "train+eval", "time": time_taken}
        return (
            {"duration": duration_dict, **train_run_stats},
            {"duration": duration_dict, **eval_run_stats},
        )

    def export_statistics(
        self,
        dataset_stats=None,
        steps=None,
        batch_size=None,
        train_hooks=None,
        eval_hooks=None,
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
            "effective_batch_size": batch_size,
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

    def init_estimator_if_none(self):
        if self.estimator is None:
            self.create_estimator()

    def create_estimator(self):
        myconfig = tf.estimator.RunConfig(tf_random_seed=1234)
        self.estimator = tf.estimator.Estimator(
            model_fn=self.model_fn,
            params={"feature_columns": self.feature_columns, **self.params},
            model_dir=self.model_dir,
            config=myconfig,
        )
