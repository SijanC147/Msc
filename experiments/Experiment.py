import os
import time
import inspect
import json
import tensorflow as tf

class Experiment():
    def __init__(self, dataset, embedding, model, make_repeatable=True, continue_training=False, custom_tag=''):
        """Create a new experiment
        
        Arguments:
            dataset {Dataset} -- dataset to use for the experiment
            embedding {Embedding} -- embedding to use for the experiment
            model {Model} -- model to use for the experiment
        
        Keyword Arguments:
            make_repeatable {bool} -- whether to set a seed for reproduceable results (default: {True})
            continue_training {bool} -- whether this is a continuation on a previous model training (default: {False})
            custom_tag {str} -- a custom string that can be appended to the experiment directory (default: {''})
        """
        if make_repeatable:
            tf.set_random_seed(1)
        self.embedding = embedding
        self.dataset = dataset
        self.model = model
        self.experiment_directory = self.init_experiment_directory(custom_tag, continue_training)
        self.tb_summary_directory = os.path.join(self.experiment_directory, 'tb_summary')
        self.init_dataset()
        self.init_model()

    def init_experiment_directory(self, custom_tag, continue_training):
        """Initialize all the directory names for the experiment
        
        Arguments:
            custom_tag {str} -- a custome string that can be appended to the experiment folder
            continue_training {bool} -- wether this is a continuation of previous training, so as to use the same directory
        
        Returns:
            str -- a path to the experiment directory
        """
        all_experiments_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
        relative_model_path = os.path.join(os.path.relpath(os.path.dirname(inspect.getfile(self.model.__class__)), os.path.join(os.getcwd(), 'models')), self.model.__class__.__name__)
        if len(custom_tag)>0:
            experiment_folder_name = '_'.join([self.dataset.__class__.__name__,self.embedding.__class__.__name__,self.embedding.get_alias(),self.embedding.get_version(),custom_tag.replace(" ", "_")])
        else:
            experiment_folder_name = '_'.join([self.dataset.__class__.__name__,self.embedding.__class__.__name__,self.embedding.get_alias(),self.embedding.get_version()])
        experiment_directory = os.path.join(all_experiments_path, relative_model_path, experiment_folder_name)
        if os.path.exists(experiment_directory) and not(continue_training):
            i = 0
            while os.path.exists(experiment_directory):
                i += 1
                experiment_directory = os.path.join(all_experiments_path, relative_model_path, experiment_folder_name+'_'+str(i))
        return experiment_directory

    def init_dataset(self):
        """Attach the embedding to the dataset
        """
        self.dataset.initialize_with_embedding(self.embedding)

    def init_model(self):
        """Call function on the model to attach the embedding and dataset, set the tensorboard summary dir, and initialize internal defaults
        """
        self.model.initialize_with_external_sources(embedding=self.embedding, dataset=self.dataset, model_dir=self.tb_summary_directory) 

    def open_tensorboard(self, debug=False):
        """Helper function to automatically launch tensorboard when an experiment is finished

        The function has to open the webpage first before starting the tensorboard process. 
        The page will appear not to load at first, but should refresh once the tensorboard process is up and running.
        
        Keyword Arguments:
            debug {bool} -- wether to start the tensorflow debugger service as well (default: {False})
        """
        os.system("open http://localhost:6006")
        if debug:
            os.system("tensorboard --logdir {0} --debugger_port 6064".format(self.tb_summary_directory))
        else:
            os.system("tensorboard --logdir {0}".format(self.tb_summary_directory))

    def run(self, mode, steps, batch_size=None, train_hooks=None, eval_hooks=None, debug=False, open_tensorboard=False):
        """Run a particular job on an experiment.
        
        Arguments:
            mode {'train','eval','train+eval'} -- identifies the particular job to run
            steps {int} -- number of steps to train for
        
        Keyword Arguments:
            batch_size {int} -- batch size to use, defaults to value in model params (default: {None})
            train_hooks {list} -- any hooks to attach to the training process (default: {None})
            eval_hooks {list} -- any hooks to attach to the evaluation process (default: {None})
            debug {bool} -- flag that is passed forward to load data from a debug dataset AND to launch tensorboard debugger if open_tensorboard is True (default: {False})
            open_tensorboard {bool} -- wether to start and open the tensorboard service after the experiment is finished (default: {False})
        """
        if mode=='train':
            train_stats = self.model.train(steps, batch_size, train_hooks, debug)
            self.write_stats_to_experiment_dir(run='train', run_stats=train_stats)
        elif mode=='eval':
            eval_stats = self.model.evaluate(eval_hooks)
            self.write_stats_to_experiment_dir(run='eval', run_stats=eval_stats)
        elif mode=='train+eval':
            train_stats, eval_stats = self.model.train_and_evaluate(steps=steps, batch_size=batch_size, train_hooks=train_hooks, eval_hooks=eval_hooks)
            self.write_stats_to_experiment_dir(run='train', run_stats=train_stats)
            self.write_stats_to_experiment_dir(run='eval', run_stats=eval_stats)

        if open_tensorboard:
            self.open_tensorboard(debug=debug)
        
    def write_stats_to_experiment_dir(self, run, run_stats):
        """Write the statistics from a particular tun to the experiment directory
        
        Arguments:
            run {'train','eval'} -- specifies different directories for the two processes for better organisation
            run_stats {dict} -- the dictionary of statistics from a run that are to be written 
        """
        run_stats_directory = os.path.join(self.experiment_directory, run)
        os.makedirs(run_stats_directory, exist_ok=True) 
        with open(os.path.join(run_stats_directory, 'dataset.json'), 'w') as file:
            file.write(json.dumps(run_stats['dataset']))
        with open(os.path.join(run_stats_directory, 'run.json'), 'w') as file:
            file.write(json.dumps({
                'duration':run_stats['duration'],
                'steps':run_stats['steps'],
                'effective_batch_size':run_stats['effective_batch_size']}))
        with open(os.path.join(run_stats_directory, 'model.md'), 'w') as file:
            file.write('## Model Params\n')
            file.write('````Python\n')
            file.write(str(run_stats['model']['params'])+'\n')
            file.write('````\n')
            file.write('## Train Input Fn\n')
            file.write('````Python\n')
            file.write(str(run_stats['model']['train_input_fn'])+'\n')
            file.write('````\n')
            file.write('## Eval Input Fn\n')
            file.write('````Python\n')
            file.write(str(run_stats['model']['eval_input_fn'])+'\n')
            file.write('````\n')
            file.write('## Model Fn\n')
            file.write('````Python\n')
            file.write(str(run_stats['model']['model_fn'])+'\n')
            file.write('````\n')
        with open(os.path.join(run_stats_directory, 'estimator.md'), 'w') as file:
            file.write('## Train Hooks\n')
            file.write('````Python\n')
            file.write(str(run_stats['estimator']['train_hooks'])+'\n')
            file.write('````\n')
            file.write('## Eval Hooks\n')
            file.write('````Python\n')
            file.write(str(run_stats['estimator']['eval_hooks'])+'\n')
            file.write('````\n')
            file.write('## Train Fn\n')
            file.write('````Python\n')
            file.write(str(run_stats['estimator']['train_fn'])+'\n')
            file.write('````\n')
            file.write('## Eval Fn\n')
            file.write('````Python\n')
            file.write(str(run_stats['estimator']['eval_fn'])+'\n')
            file.write('````\n')
            file.write('## Train And Eval Fn\n')
            file.write('````Python\n')
            file.write(str(run_stats['estimator']['train_eval_fn'])+'\n')
            file.write('````\n')
        if (len(run_stats['common'])>0):
            with open(os.path.join(run_stats_directory, 'common.md'), 'w') as file:
                file.write('## Model Common Functions\n')
                file.write('````Python\n')
                file.write(str(run_stats['common'])+'\n')
                file.write('````\n')

