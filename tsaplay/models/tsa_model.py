from abc import ABC, abstractmethod
import comet_ml
import tensorflow as tf
from tensorflow.python import debug as tf_debug  # pylint: disable=E0611
from tensorflow.estimator import (  # pylint: disable=E0401
    RunConfig,
    Estimator,
    ModeKeys,
    EstimatorSpec,
)
from tensorflow.estimator.export import (  # pylint: disable=E0401
    ServingInputReceiver,
)
from tsaplay.utils.tf import (
    make_input_fn,
    ids_lookup_table,
    make_dense_features,
    embed_sequences,
    sharded_saver,
)
from tsaplay.utils.comet import cometml, log_dist_data, log_features_asset_data
from tsaplay.utils.addons import (
    addon,
    prediction_outputs,
    conf_matrix,
    f1_scores,
    logging,
    histograms,
    scalars,
    metadata,
)
from tsaplay.utils.debug import cprnt


class TsaModel(ABC):
    def __init__(self, params=None, aux_config=None, run_config=None):
        self._comet_experiment = None
        self._estimator = None
        self.aux_config = aux_config or {}
        self._hooks = (
            []
            if not self.aux_config.get("debug")
            else [tf_debug.LocalCLIDebugHook()]
            if self.aux_config.get("debug") == "cli"
            else [
                tf_debug.TensorBoardDebugHook(
                    "localhost:{}".format(self.aux_config.get("debug"))
                )
            ]
        )
        self.run_config = RunConfig(**(run_config or {}))
        self.params = self.set_params()
        if params:
            self.params.update(params)

    @property
    def name(self):
        return self.__class__.__name__

    @property
    def comet_experiment(self):
        return self._comet_experiment

    @property
    def estimator(self):
        return self._estimator

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
    def train_input_fn(cls, tfrecords, params):
        pass

    @classmethod
    @make_input_fn("EVAL")
    def eval_input_fn(cls, tfrecords, params):
        pass

    @classmethod
    def make_estimator_spec(cls, mode, logits, optimizer, loss):
        predictions = {
            "class_ids": tf.argmax(logits, 1),
            "probabilities": tf.nn.softmax(logits),
            "logits": logits,
        }
        if mode == ModeKeys.PREDICT:
            return EstimatorSpec(mode, predictions=predictions)

        if mode == ModeKeys.EVAL:
            return EstimatorSpec(mode, predictions=predictions, loss=loss)

        train_op = optimizer.minimize(
            loss, global_step=tf.train.get_global_step()
        )

        return EstimatorSpec(
            mode, loss=loss, train_op=train_op, predictions=predictions
        )

    def train(self, feature_provider, steps):
        log_dist_data(self.comet_experiment, feature_provider, ["train"])
        log_features_asset_data(self.comet_experiment, feature_provider)
        self._initialize_estimator(feature_provider)
        self._estimator.train(
            input_fn=lambda: self.train_input_fn(
                tfrecords=feature_provider.train_tfrecords, params=self.params
            ),
            steps=steps,
            hooks=self._hooks,
        )

    def evaluate(self, feature_provider):
        log_dist_data(self.comet_experiment, feature_provider, ["test"])
        log_features_asset_data(self.comet_experiment, feature_provider)
        self._initialize_estimator(feature_provider)
        self._estimator.evaluate(
            input_fn=lambda: self.eval_input_fn(
                tfrecords=feature_provider.test_tfrecords, params=self.params
            ),
            hooks=self._hooks,
        )

    def train_and_eval(self, feature_provider, steps):
        log_dist_data(
            self.comet_experiment, feature_provider, ["train", "test"]
        )
        log_features_asset_data(self.comet_experiment, feature_provider)
        self._initialize_estimator(feature_provider)
        train_spec = tf.estimator.TrainSpec(
            input_fn=lambda: self.train_input_fn(
                tfrecords=feature_provider.train_tfrecords, params=self.params
            ),
            max_steps=steps,
            hooks=self._hooks,
        )
        eval_spec = tf.estimator.EvalSpec(
            input_fn=lambda: self.eval_input_fn(
                tfrecords=feature_provider.test_tfrecords, params=self.params
            ),
            steps=None,
            throttle_secs=0,
            hooks=self._hooks,
        )
        tf.estimator.train_and_evaluate(
            estimator=self._estimator,
            train_spec=train_spec,
            eval_spec=eval_spec,
        )

    def export(self, directory, feature_provider):
        self._initialize_estimator(feature_provider)
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

        # TODO: Why does this function call not have a value for number of OOV buckets?
        ids_table = ids_lookup_table(
            self.params["_vocab_file"],
            oov_buckets=self.params["_num_oov_buckets"],
        )
        features = {
            "left": parsed_example["left"],
            "target": parsed_example["target"],
            "right": parsed_example["right"],
            "left_ids": ids_table.lookup(parsed_example["left"]),
            "target_ids": ids_table.lookup(parsed_example["target"]),
            "right_ids": ids_table.lookup(parsed_example["right"]),
        }

        sparse_features = self.processing_fn(features)
        input_features = make_dense_features(sparse_features)

        inputs = {"instances": inputs_serialized}

        return ServingInputReceiver(input_features, inputs)

    @cometml
    @sharded_saver
    @addon([scalars, logging, metadata, histograms, conf_matrix, f1_scores])
    @addon([prediction_outputs])
    @embed_sequences
    def _model_fn(self, features, labels, mode, params):
        return self.model_fn(features, labels, mode, params)

    def _initialize_estimator(self, feature_provider):
        self.aux_config["_feature_provider"] = feature_provider.name
        if not self.aux_config.get("class_labels"):
            class_labels = map(str, sorted(feature_provider.class_labels))
            self.aux_config["class_labels"] = list(class_labels)
        self.params["_n_out_classes"] = len(self.aux_config["class_labels"])
        self.params.update(feature_provider.embedding_params)
        self._estimator = Estimator(
            model_fn=self._model_fn, params=self.params, config=self.run_config
        )
