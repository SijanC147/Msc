import os
import tensorflow as tf
import inspect
import time
import datetime
from utils import get_statistics_on_features_labels,change_features_labels_distribution
from abc import ABC, abstractmethod

class Model(ABC):
    def __init__(self, embedding=None, dataset=None, model_dir=None):
        """Create a new model
        
        Arguments:
            ABC {Class} -- Make the class abstract
        
        Keyword Arguments:
            embedding {Embedding} -- Specify an embedding to use with the model, can be defined later (default: {None})
            dataset {Dataset} -- Specify a dataset to use with the model, can be defined later (default: {None})
            model_dir {str} -- Directory for the tensorboard summary files, can be defined later (default: {None})
        """
        self.connect_external_sources(embedding=embedding, dataset=dataset, model_dir=model_dir)
        self.estimator = None

    @abstractmethod
    def set_params(self, params):
        """Set params value for the model.
        
        Must ALWAYS be called at the end of child-classes overrides to set the params
        
        Arguments:
            params {dict} -- dictionary of params to be passed to the model
        """
        self.params = {
            'feature_columns': self.feature_columns, 
            'embedding_initializer': self.embedding.get_tf_embedding_initializer(),
            'vocab_size': self.embedding.get_vocab_size(),
            'embedding_dim': self.embedding.get_embedding_dim(),
            **params}

    @abstractmethod
    def set_feature_columns(self, feature_columns):
        """Set feature columns for the model.
        
        Must ALWAYS be called at the end of child-classes overrides to set the feature colums
        
        Arguments:
            feature_columns {list} -- list of tensorflow feature columns for the model
        """
        self.feature_columns = feature_columns
    
    @abstractmethod
    def set_train_input_fn(self, train_input_fn):
        """Set the training input function for the model.
        
        Must ALWAYS be called at the end of child-classes overrides to set the train_input_fn for the model 
        
        Arguments:
            train_input_fn {callable} -- function that provides data that is called when training the model
        """
        self.train_input_fn = train_input_fn
    
    @abstractmethod
    def set_eval_input_fn(self, eval_input_fn):
        """Set the evaluation input function for the model.
        
        Must ALWAYS be called at the end of child-classes overrides to set the eval_input_fn for the model 
        
        Arguments:
            eval_input_fn {callable} -- function that provides data that is called when evaluating the model
        """
        self.eval_input_fn = eval_input_fn

    @abstractmethod
    def set_model_fn(self, model_fn):
        """Set the specific model function for the model
        
        Must ALWAYS be called at the end of child-classes overrides to set the model function        

        Arguments:
            model_fn {callable} -- a model_fn defining the internals of the model itself, as per TF specifications
        """
        self.model_fn = model_fn

    def initialize_with_external_sources(self, embedding, dataset, model_dir):
        """Attach an embedding, dataset(with matching embedding) and model_dir for the model, and initialize internals of the model
        
        Arguments:
            embedding {Embedding} -- embedding to attach to the model, must match the dataset embedding
            dataset {Dataset} -- dataset to attach to the model
            model_dir {str} -- path where tensorboard summaries are written to for the model
        """
        self.connect_external_sources(embedding, dataset, model_dir)
        self.initialize_internal_defaults()
    
    def connect_external_sources(self, embedding, dataset, model_dir):
        """Attach an embedding, dataset(with matching embedding) and model_dir for the model

        Does NOT initialize the internals of the model
        
        Arguments:
            embedding {Embedding} -- embedding to attach to the model, must match the dataset embedding
            dataset {Dataset} -- dataset to attach to the model
            model_dir {str} -- path where tensorboard summaries are written to for the model
        """
        self.set_embedding(embedding)
        self.set_dataset(dataset)
        self.set_model_dir(model_dir)

    def set_embedding(self, embedding):
        """Attach an embedding to the model

        Must match the attached dataset's embedding
        
        Arguments:
            embedding {Embedding} -- embedding to attach to the model
        """
        self.embedding = embedding

    def set_dataset(self, dataset):
        """Attach a dataset to the model

        Dataset must have an embedding that matches the attached embedding to this model
        
        Arguments:
            dataset {Dataset} -- the dataset to attach to the model
        """ 
        self.dataset = dataset

    def set_model_dir(self, model_dir):
        """Specify the tensorboard summary directory for the model
        
        Arguments:
            model_dir {str} -- directory to save tensorboard summary files
        """
        self.model_dir = model_dir

    def init_estimator_if_none(self):
        """Initialize the estimator object if it has not yet been initialized
        """
        if self.estimator==None:
            self.create_estimator()

    def create_estimator(self):
        """Create a new estimator object for this model
        
        The internal settings for the model MUST have been initialized before this is called
        either to their default, or the setter respective functions
        """
        myconfig = tf.estimator.RunConfig(tf_random_seed=1234)
        self.estimator = tf.estimator.Estimator(
            model_fn=self.model_fn,
            params={'feature_columns': self.feature_columns, **self.params},
            model_dir=self.model_dir,
            config=myconfig
        )

    def train(self, steps, batch_size=None, hooks=None, debug=False, label_distribution=None):
        """Run a training job on the model estimator
        
        Arguments:
            steps {int} -- number of steps to train for
        
        Keyword Arguments:
            batch_size {int} -- batch size to be used, otherwise loaded from the model params (default: {None})
            hooks {list} -- any training hooks to attach (default: {None})
            debug {bool} -- load features and labels from a debug dataset (default: {False})
        
        Returns:
            dict -- dictionary of statistics obtained from the training process
        """
        batch = batch_size if batch_size!=None else self.params['batch_size']
        if not(debug):
            features, labels = self.get_features_and_labels(mode='train', label_distribution=label_distribution)
        else:
            features, labels = self.get_features_and_labels(mode='debug', label_distribution=label_distribution)
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

    def evaluate(self, hooks=None, debug=False, label_distribution=None):
        """Run an evaluation job on the model estimator, useless if the model has not been trained
        
        Keyword Arguments:
            hooks {list} -- any hooks to pass to attach (default: {None})
            debug {bool} -- load features and labels from a debug dataset (default: {False})
        
        Returns:
            dict -- dictionary of statistics obtained from the evaaluation process
        """
        if not(debug):
            features, labels = self.get_features_and_labels(mode='eval', label_distribution=label_distribution)
        else:
            features, labels = self.get_features_and_labels(mode='debug', label_distribution=label_distribution)
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

    def train_and_evaluate(self, steps=None, batch_size=None, train_hooks=None, eval_hooks=None, train_distribution=None, eval_distribution=None):
        """Run a training job followed by an evaluation job on the model estimator
        
        Keyword Arguments:
            steps {int} -- number of steps to train for (default: {None})
            batch_size {int} -- batch size to use for training, defaults to value in model params (default: {None})
            train_hooks {list} -- any hooks to attach to the training job (default: {None})
            eval_hooks {list} -- any hooks to attach to the evaluation job (default: {None})
        
        Returns:
            (dict, dict) -- two separate dictionaries of statistics with training, and evaluation statistics
        """

        if batch_size!=None and batch_size!=self.params['batch_size']:
            self.params['batch_size']=batch_size

        train_features, train_labels = self.get_features_and_labels(mode='train', label_distribution=train_distribution)
        eval_features, eval_labels = self.get_features_and_labels(mode='eval', label_distribution=eval_distribution)
        self.init_estimator_if_none()
        train_run_stats = self.export_statistics(features=train_features,labels=train_labels, batch_size=batch_size, steps=steps, train_hooks=train_hooks, eval_hooks=eval_hooks)
        eval_run_stats = self.export_statistics(features=eval_features,labels=eval_labels, batch_size=batch_size, steps=steps, train_hooks=train_hooks, eval_hooks=eval_hooks)
        # os.makedirs(self.estimator.eval_dir(), exist_ok=True)
        # early_stopping = tf.contrib.estimator.stop_if_no_decrease_hook(
        #     self.estimator,
        #     metric_name='loss',
        #     max_steps_without_decrease=10,
        #     min_steps=300)
        print("{0} starting to train and evaluate...".format(self.__class__.__name__))
        start = time.time()
        tf.estimator.train_and_evaluate(
            estimator=self.estimator,
            train_spec=tf.estimator.TrainSpec(
                input_fn = lambda: self.train_input_fn(
                    features=train_features,
                    labels=train_labels,
                    batch_size=self.params['batch_size']
                ),
                max_steps=steps,
                # hooks=[early_stopping] if train_hooks==None else [early_stopping]+train_hooks
                hooks=train_hooks
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
        """Generate a dictionary of different statistics to be returned after estimator job runs
        
        Arguments:
            features {dict} -- dictionary of features fed to the model
            labels {list} -- list of corresponsing labels fed to the model
        
        Keyword Arguments:
            steps {int} -- the number of steps used in training (default: {None})
            batch_size {int} -- the batch size used (default: {None})
            train_hooks {list} -- any training hooks that were attached (default: {None})
            eval_hooks {list} -- any evaluation hooks that were attached (default: {None})
        
        Returns:
            dict -- dictionary of the statistics to record about a job
        """
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
        """Initialize the internal settings of the model to the default values
        
        Default values are specified in the respective setter function of the sub-class model
        """
        self.set_feature_columns(None)
        self.set_params(None)
        self.set_train_input_fn(None)
        self.set_eval_input_fn(None)
        self.set_model_fn(None) 

    def get_features_and_labels(self, mode, label_distribution=None):
        """Get features and labels from the dataset for a particular mode
        
        Arguments:
            mode {'train','eval','debug} -- specifies the source file to get the features and labels from
        
        Returns:
            (dict, list) -- dictionary of features and list of corresponding labels
        """
        features, labels = self.dataset.get_mapped_features_and_labels(mode)
        if label_distribution!=None:
            features, labels = change_features_labels_distribution(features, labels, label_distribution)
        return features, labels
    