from abc import ABC, abstractmethod
import comet_ml
import tensorflow as tf
from tensorflow.estimator import RunConfig, Estimator  # pylint: disable=E0401
from tensorflow.estimator.export import (  # pylint: disable=E0401
    ServingInputReceiver
)
from tsaplay.features.FeatureProvider import FeatureProvider
from tsaplay.utils.decorators import make_input_fn, addon, prepare
from tsaplay.models.addons import (
    embeddings,
    cometml,
    export_outputs,
    conf_matrix,
    logging,
    histograms,
    scalars,
)


class TSAModel(ABC):
    def __init__(self, params=None, config=None):
        self._comet_experiment = None
        self._estimator = None
        self.class_labels = ["Negative", "Neutral", "Positive"]
        self.run_config = RunConfig(**(config or {}))
        self.params = self.set_params()
        if params is not None:
            self.params.update(params)

    @property
    def name(self):
        return self.__class__.__name__

    @property
    def comet_experiment(self):
        return self._comet_experiment

    @abstractmethod
    def set_params(self):
        pass

    @classmethod
    def processing_fn(cls, features):
        return features

    @abstractmethod
    def model_fn(self, features, labels, mode, params):
        pass

    def attach_comet_ml_experiment(self, api_key, exp_key):
        self._comet_experiment = comet_ml.ExistingExperiment(
            api_key=api_key, previous_experiment=exp_key
        )

    @classmethod
    @make_input_fn("TRAIN")
    def train_input_fn(cls, tfrecord, batch_size):
        pass

    @classmethod
    @make_input_fn("EVAL")
    def eval_input_fn(cls, tfrecord, batch_size):
        pass

    def train(self, feature_provider, steps):
        self._initialize_estimator(feature_provider.embedding_params)
        self._estimator.train(
            input_fn=lambda: self.train_input_fn(
                tfrecord=feature_provider.train_tfrecords,
                batch_size=self.params["batch_size"],
            ),
            steps=steps,
        )

    def evaluate(self, feature_provider):
        self._initialize_estimator(feature_provider.embedding_params)
        self._estimator.evaluate(
            input_fn=lambda: self.eval_input_fn(
                tfrecord=feature_provider.test_tfrecords,
                batch_size=self.params["batch_size"],
            )
        )

    def train_and_eval(self, feature_provider, steps):
        self._initialize_estimator(feature_provider.embedding_params)
        train_spec = tf.estimator.TrainSpec(
            input_fn=lambda: self.train_input_fn(
                tfrecord=feature_provider.train_tfrecords,
                batch_size=self.params["batch_size"],
            ),
            max_steps=steps,
        )
        eval_spec = tf.estimator.EvalSpec(
            input_fn=lambda: self.eval_input_fn(
                tfrecord=feature_provider.test_tfrecords,
                batch_size=self.params["batch_size"],
            ),
            steps=None,
        )
        tf.estimator.train_and_evaluate(
            estimator=self._estimator,
            train_spec=train_spec,
            eval_spec=eval_spec,
        )

    def export(self, directory, embedding_params):
        self._initialize_estimator(embedding_params)
        self._estimator.export_savedmodel(
            export_dir_base=directory,
            serving_input_receiver_fn=self._serving_input_receiver_fn,
            strip_default_attrs=True,
        )

    def _serving_input_receiver_fn(self):
        inputs_serialized = tf.placeholder(dtype=tf.string)
        feature_spec = {
            "left": tf.VarLenFeature(dtype=tf.string),
            "target": tf.VarLenFeature(dtype=tf.string),
            "right": tf.VarLenFeature(dtype=tf.string),
        }
        parsed_example = tf.parse_example(inputs_serialized, feature_spec)

        ids_table = FeatureProvider.index_lookup_table(
            self.params["vocab_file_path"]
        )
        features = {
            "left": parsed_example["left"],
            "target": parsed_example["target"],
            "right": parsed_example["right"],
            "left_ids": ids_table.lookup(parsed_example["left"]),
            "target_ids": ids_table.lookup(parsed_example["target"]),
            "right_ids": ids_table.lookup(parsed_example["right"]),
        }

        input_features = self.processing_fn(features)

        inputs = {"instances": inputs_serialized}

        return ServingInputReceiver(input_features, inputs)

    @prepare([cometml, embeddings])
    @addon([scalars, logging, histograms, conf_matrix])
    @addon([export_outputs])
    def _model_fn(self, features, labels, mode, params):
        return self.model_fn(features, labels, mode, params)

    def _initialize_estimator(self, embedding_params):
        self.params.update(embedding_params)
        self._estimator = Estimator(
            model_fn=self._model_fn, params=self.params, config=self.run_config
        )