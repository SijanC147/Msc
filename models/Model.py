import os
import tensorflow as tf
import inspect
from abc import ABC, abstractmethod

class Model(ABC):
    def __init__(self, experiment_name, source_dataset):
        self.dataset = source_dataset
        self.embedding = self.dataset.embedding
        self.experiment_log_dir = self.get_experiment_log_directory(experiment_name)
        self.estimator = None
        self.initialize_defaults_for_model()

    @abstractmethod
    def set_params(self, params):
        self.params = params

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

    def train(self, steps, batch_size=None, hooks=None):
        batch = batch_size if batch_size!=None else self.params['batch_size']
        features, labels = self.get_features_and_labels(mode='train')
        if self.estimator==None:
            self.create_estimator()
        self.estimator.train(
            input_fn = lambda: self.train_input_fn(
                features=features,
                labels=labels,
                batch_size=batch
            ),
            steps=steps,
            hooks=hooks
        )

    def evaluate(self, batch_size=None, hooks=None):
        batch = batch_size if batch_size!=None else self.params['batch_size']
        features, labels = self.get_features_and_labels(mode='eval')
        if self.estimator==None:
            self.create_estimator()
        self.estimator.evaluate(
            input_fn = lambda: self.eval_input_fn(
                features=features,
                labels=labels,
                batch_size=batch
            ),
            hooks=hooks
        )

    def initialize_defaults_for_model(self):
        self.set_params(None)
        self.set_feature_columns(None)
        self.set_train_input_fn(None)
        self.set_eval_input_fn(None)
        self.set_model_fn(None) 

    def get_experiment_log_directory(self, experiment_name):
        experiment_log_parent_folder = os.path.join(os.getcwd(), 'logs', os.path.relpath(os.path.dirname(inspect.getfile(self.__class__)), os.path.join(os.getcwd(), 'models')), self.__class__.__name__)
        return os.path.join(experiment_log_parent_folder, experiment_name.replace(" ", "_")) 

    def get_features_and_labels(self, mode):
        return self.dataset.get_mapped_features_and_labels(mode)
    
    def create_estimator(self):
        self.estimator = tf.estimator.Estimator(
            model_fn=self.model_fn,
            params={'feature_columns': self.feature_columns, **self.params},
            model_dir=self.experiment_log_dir
        )