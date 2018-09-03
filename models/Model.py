import os
import tensorflow as tf
import inspect
from abc import ABC, abstractmethod

class Model(ABC):
    def __init__(self, embedding=None, dataset=None, model_dir=None):
        self.connect_external_sources(embedding=embedding, dataset=dataset, model_dir=model_dir)
        self.estimator = None

    @abstractmethod
    def set_params(self, params):
        self.params = {'feature_columns': self.feature_columns, **params}

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

    def initialize_with_external_sources(self, embedding, dataset, model_dir):
        self.connect_external_sources(embedding, dataset, model_dir)
        self.initialize_internal_defaults()
    
    def connect_external_sources(self, embedding, dataset, model_dir):
        self.set_embedding(embedding)
        self.set_dataset(dataset)
        self.set_model_dir(model_dir)

    def set_embedding(self, embedding):
        self.embedding = embedding

    def set_dataset(self, dataset):
        self.dataset = dataset

    def set_model_dir(self, model_dir):
        self.model_dir = model_dir

    def init_estimator_if_none(self):
        if self.estimator==None:
            self.create_estimator()

    def train(self, steps, batch_size=None, hooks=None, debug=False):
        batch = batch_size if batch_size!=None else self.params['batch_size']
        if not(debug):
            features, labels = self.get_features_and_labels(mode='train')
        else:
            features, labels = self.get_features_and_labels(mode='debug')
        self.init_estimator_if_none()
        self.estimator.train(
            input_fn = lambda: self.train_input_fn(
                features=features,
                labels=labels,
                batch_size=batch
            ),
            steps=steps,
            hooks=hooks
        )

    def evaluate(self, hooks=None, debug=False):
        if not(debug):
            features, labels = self.get_features_and_labels(mode='eval')
        else:
            features, labels = self.get_features_and_labels(mode='debug')
        self.init_estimator_if_none()
        self.estimator.evaluate(
            input_fn = lambda: self.eval_input_fn(
                features=features,
                labels=labels
            ),
            hooks=hooks
        )

    def train_and_evaluate(self, steps=None, batch_size=None, train_hooks=None, eval_hooks=None):
        batch = batch_size if batch_size!=None else self.params['batch_size']
        train_features, train_labels = self.get_features_and_labels(mode='train')
        eval_features, eval_labels = self.get_features_and_labels(mode='eval')
        self.init_estimator_if_none()
        os.makedirs(self.estimator.eval_dir(), exist_ok=True)
        early_stopping = tf.contrib.estimator.stop_if_no_decrease_hook(
            self.estimator,
            metric_name='loss',
            max_steps_without_decrease=10,
            min_steps=300)
        tf.estimator.train_and_evaluate(
            estimator=self.estimator,
            train_spec=tf.estimator.TrainSpec(
                input_fn = lambda: self.train_input_fn(
                    features=train_features,
                    labels=train_labels,
                    batch_size=batch
                ),
                max_steps=steps,
                hooks=[early_stopping] if train_hooks==None else [early_stopping]+train_hooks
            ),
            eval_spec=tf.estimator.EvalSpec(
                input_fn = lambda: self.eval_input_fn(
                    features=eval_features,
                    labels=eval_labels
                ),
                steps=None,
                hooks=eval_hooks
            )
        )

    def initialize_internal_defaults(self):
        self.set_feature_columns(None)
        self.set_params(None)
        self.set_train_input_fn(None)
        self.set_eval_input_fn(None)
        self.set_model_fn(None) 

    def get_features_and_labels(self, mode):
        return self.dataset.get_mapped_features_and_labels(mode)
    
    def create_estimator(self):
        self.estimator = tf.estimator.Estimator(
            model_fn=self.model_fn,
            params={'feature_columns': self.feature_columns, **self.params},
            model_dir=self.model_dir
        )