import tensorflow as tf
from abc import ABC, abstractmethod
from tsaplay.utils.data import make_input_fn
from tsaplay.utils.decorators import attach_embedding_params


class SlimModel(ABC):
    def __init__(self, params, run_config=None):
        self.params = params
        self.run_config = run_config

    @property
    def name(self):
        return self.__class__.__name__

    @classmethod
    @abstractmethod
    def process_features(cls, features):
        pass

    @classmethod
    @abstractmethod
    def model_fn(cls, features, labels, mode, params):
        pass

    @property
    def estimator(self):
        return tf.estimator.Estimator(
            model_fn=self.model_fn, params=self.params, config=self.run_config
        )

    @classmethod
    @make_input_fn("TRAIN")
    def train_input_fn(cls, tfrecord, batch_size):
        pass

    @classmethod
    @make_input_fn("EVAL")
    def eval_input_fn(cls, tfrecord, batch_size):
        pass

    @classmethod
    @make_input_fn("PREDICT")
    def serving_input_fn(cls, features):
        pass

    @attach_embedding_params
    def train(self, feature_provider, steps):
        self.estimator.train(
            input_fn=lambda: self.train_input_fn(
                tfrecord=feature_provider.train_tfrecords,
                batch_size=self.params["batch_size"],
            ),
            steps=steps,
            # hooks=self._attach_std_train_hooks(self.train_hooks),
        )

    @attach_embedding_params
    def evaluate(self, feature_provider):
        self.estimator.evaluate(
            input_fn=lambda: self.eval_input_fn(
                tfrecord=feature_provider.test_tfrecords,
                batch_size=self.params["batch_size"],
            ),
            # hooks=self._attach_std_eval_hooks(self.eval_hooks),
        )

    @attach_embedding_params
    def train_and_eval(self, feature_provider, steps):
        # stop_early = self.params.get("early_stopping", False)
        train_spec = tf.estimator.TrainSpec(
            input_fn=lambda: self.train_input_fn(
                tfrecord=feature_provider.train_tfrecords,
                batch_size=self.params["batch_size"],
            ),
            max_steps=steps,
            # hooks=self._attach_std_train_hooks(self.train_hooks, stop_early),
        )
        eval_spec = tf.estimator.EvalSpec(
            input_fn=lambda: self.eval_input_fn(
                tfrecord=feature_provider.test_tfrecords,
                batch_size=self.params["batch_size"],
            ),
            steps=None,
            # hooks=self._attach_std_eval_hooks(self.eval_hooks),
        )
        tf.estimator.train_and_evaluate(
            estimator=self.estimator,
            train_spec=train_spec,
            eval_spec=eval_spec,
        )

    def _processing_fn(self, features, label=None):
        processed_features = self.process_features(features)
        if label is None:
            return processed_features
        else:
            return (processed_features, label)
