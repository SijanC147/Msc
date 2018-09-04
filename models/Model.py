import os
import tensorflow as tf
import inspect
import time
import datetime
from utils import get_statistics_on_features_labels,change_features_labels_distribution
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
        run_stats = self.export_statistics(features=features,labels=labels, batch_size=batch_size, steps=steps, train_hooks=hooks)
        print("Training {0}...".format(self.__class__.__name__))
        start = time.time()
        self.estimator.train(
            input_fn = lambda: self.train_input_fn(
                features=features,
                labels=labels,
                batch_size=batch
            ),
            steps=steps,
            hooks=hooks
        )
        time_taken = str(datetime.timedelta(seconds=time.time()-start))
        duration_dict = {
            'job': 'train',
            'time': time_taken 
        }
        print("Finished training {0} in {1}".format(self.__class__.__name__, time_taken))
        return {'duration':duration_dict, **run_stats}

    def evaluate(self, hooks=None, debug=False):
        if not(debug):
            features, labels = self.get_features_and_labels(mode='eval')
        else:
            features, labels = self.get_features_and_labels(mode='debug')
        self.init_estimator_if_none()
        run_stats = self.export_statistics(features=features,labels=labels, eval_hooks=hooks)
        print("Evaluating {0}...".format(self.__class__.__name__))
        start = time.time()
        self.estimator.evaluate(
            input_fn = lambda: self.eval_input_fn(
                features=features,
                labels=labels
            ),
            hooks=hooks
        )
        time_taken = str(datetime.timedelta(seconds=time.time()-start))
        print("Finished evaluating {0} in {1}".format(self.__class__.__name__, time_taken))
        duration_dict = {
            'job': 'eval',
            'time': time_taken 
        }
        return {'duration':duration_dict, **run_stats}

    def train_and_evaluate(self, steps=None, batch_size=None, train_hooks=None, eval_hooks=None):
        batch = batch_size if batch_size!=None else self.params['batch_size']
        train_features, train_labels = self.get_features_and_labels(mode='train')
        train_features, train_labels = change_features_labels_distribution(features=train_features, labels=train_labels, positive=0.20, neutral=0.70, negative=0.10)
        eval_features, eval_labels = self.get_features_and_labels(mode='eval')
        eval_features, eval_labels = change_features_labels_distribution(features=eval_features, labels=eval_labels, positive=0.35, neutral=0.20, negative=0.45)
        self.init_estimator_if_none()
        train_run_stats = self.export_statistics(features=train_features,labels=train_labels, batch_size=batch_size, steps=steps, train_hooks=train_hooks, eval_hooks=eval_hooks)
        eval_run_stats = self.export_statistics(features=eval_features,labels=eval_labels, batch_size=batch_size, steps=steps, train_hooks=train_hooks, eval_hooks=eval_hooks)
        os.makedirs(self.estimator.eval_dir(), exist_ok=True)
        early_stopping = tf.contrib.estimator.stop_if_no_decrease_hook(
            self.estimator,
            metric_name='loss',
            max_steps_without_decrease=10,
            min_steps=300)
        print("{0} starting to train and evaluate...".format(self.__class__.__name__))
        start = time.time()
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
        time_taken = str(datetime.timedelta(seconds=time.time()-start))
        print("{0} trained and evaluated in {1}".format(self.__class__.__name__, time_taken))
        duration_dict = {
            'job': 'train+eval',
            'time': time_taken 
        }
        return {'duration':duration_dict, **train_run_stats}, {'duration':duration_dict, **eval_run_stats} 

    def export_statistics(self, features, labels, steps=None, batch_size=None, train_hooks=None, eval_hooks=None):
        dataset_statistics = get_statistics_on_features_labels(features, labels)
        train_input_fn_source = inspect.getsource(self.train_input_fn)
        eval_input_fn_source = inspect.getsource(self.eval_input_fn)
        model_fn_source = inspect.getsource(self.model_fn)
        model_common_file = os.path.join(os.path.dirname(inspect.getfile(self.__class__)),'common.py')
        estimator_train_fn_source = inspect.getsource(self.train)
        estimator_eval_fn_source = inspect.getsource(self.evaluate)
        estimator_train_eval_fn_source = inspect.getsource(self.train_and_evaluate)
        if os.path.exists(model_common_file):
            common_content = open(model_common_file, 'r').read()
        else:
            common_content = ''
        return {
            'dataset' : dataset_statistics,
            'steps' : steps,
            'effective_batch_size': batch_size,
            'model': {
                'params': self.params,
                'train_input_fn': train_input_fn_source,
                'eval_input_fn': eval_input_fn_source,
                'model_fn': model_fn_source,
            },
            'estimator': {
                'train_hooks': train_hooks,
                'eval_hooks': eval_hooks,
                'train_fn': estimator_train_fn_source,
                'eval_fn': estimator_eval_fn_source,
                'train_eval_fn': estimator_train_eval_fn_source,
            },
            'common': common_content
        }
        
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