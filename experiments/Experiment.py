import os
import time
import inspect
import json
import tensorflow as tf
from utils import start_tensorboard
from embeddings.GloVe import GloVe
from datasets.Dong2014 import Dong2014


class Experiment:
    def __init__(
        self,
        dataset,
        embedding,
        model,
        run_config=None,
        seed=None,
        continue_training=False,
        custom_tag="",
        debug=False,
    ):
        if debug:
            self.embedding = GloVe(alias="twitter", version="debug")
            self.dataset = Dong2014()
        else:
            self.embedding = embedding
            self.dataset = dataset
        self.model = model
        self.experiment_directory = self.init_experiment_directory(
            custom_tag, continue_training
        )
        self.tb_summary_directory = os.path.join(
            self.experiment_directory, "tb_summary"
        )
        self.dataset.set_embedding(self.embedding)
        self.model.embedding = self.embedding
        self.model.dataset = self.dataset

        if run_config is None:
            run_config = tf.estimator.RunConfig(
                model_dir=self.tb_summary_directory
            )
        else:
            run_config = run_config.replace(
                model_dir=self.tb_summary_directory
            )
        if seed is not None:
            run_config = run_config.replace(tf_random_seed=seed)

        self.model.run_config = run_config
        self.model.initialize_internal_defaults()

    def init_experiment_directory(self, custom_tag, continue_training):
        all_experiments_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "data"
        )
        relative_model_path = os.path.join(
            os.path.relpath(
                os.path.dirname(inspect.getfile(self.model.__class__)),
                os.path.join(os.getcwd(), "models"),
            ),
            self.model.__class__.__name__,
        )
        if len(custom_tag) > 0:
            experiment_folder_name = "_".join(
                [
                    self.dataset.__class__.__name__,
                    self.embedding.__class__.__name__,
                    self.embedding.alias,
                    self.embedding.version,
                    custom_tag.replace(" ", "_"),
                ]
            )
        else:
            experiment_folder_name = "_".join(
                [
                    self.dataset.__class__.__name__,
                    self.embedding.__class__.__name__,
                    self.embedding.alias,
                    self.embedding.version,
                ]
            )
        experiment_directory = os.path.join(
            all_experiments_path, relative_model_path, experiment_folder_name
        )
        if os.path.exists(experiment_directory) and not (continue_training):
            i = 0
            while os.path.exists(experiment_directory):
                i += 1
                experiment_directory = os.path.join(
                    all_experiments_path,
                    relative_model_path,
                    experiment_folder_name + "_" + str(i),
                )
        return experiment_directory

    def run(
        self,
        job,
        steps,
        train_hooks=None,
        eval_hooks=None,
        train_distribution=None,
        eval_distribution=None,
        debug=False,
        start_tensorboard=False,
    ):
        if job == "train":
            train_stats = self.model.train(
                steps=steps,
                hooks=train_hooks,
                debug=debug,
                label_distribution=train_distribution,
            )
            self.write_stats_to_experiment_dir(
                job="train", job_stats=train_stats
            )
        elif job == "eval":
            eval_stats = self.model.evaluate(
                hooks=eval_hooks,
                debug=debug,
                label_distribution=eval_distribution,
            )
            self.write_stats_to_experiment_dir(
                job="eval", job_stats=eval_stats
            )
        elif job == "train+eval":
            train_stats, eval_stats = self.model.train_and_evaluate(
                steps=steps,
                train_hooks=train_hooks,
                eval_hooks=eval_hooks,
                train_distribution=train_distribution,
                eval_distribution=eval_distribution,
            )
            self.write_stats_to_experiment_dir(
                job="train", job_stats=train_stats
            )
            self.write_stats_to_experiment_dir(
                job="eval", job_stats=eval_stats
            )

        if start_tensorboard:
            start_tensorboard(
                summary_dir=self.tb_summary_directory, debug=debug
            )

    def write_stats_to_experiment_dir(self, job, job_stats):
        job_stats_directory = os.path.join(self.experiment_directory, job)
        os.makedirs(job_stats_directory, exist_ok=True)
        with open(
            os.path.join(job_stats_directory, "dataset.json"), "w"
        ) as file:
            file.write(json.dumps(job_stats["dataset"]))
        with open(os.path.join(job_stats_directory, "job.json"), "w") as file:
            file.write(
                json.dumps(
                    {
                        "duration": job_stats["duration"],
                        "steps": job_stats["steps"],
                    }
                )
            )
        with open(os.path.join(job_stats_directory, "model.md"), "w") as file:
            file.write("## Model Params\n")
            file.write("````Python\n")
            file.write(str(job_stats["model"]["params"]) + "\n")
            file.write("````\n")
            file.write("## Train Input Fn\n")
            file.write("````Python\n")
            file.write(str(job_stats["model"]["train_input_fn"]) + "\n")
            file.write("````\n")
            file.write("## Eval Input Fn\n")
            file.write("````Python\n")
            file.write(str(job_stats["model"]["eval_input_fn"]) + "\n")
            file.write("````\n")
            file.write("## Model Fn\n")
            file.write("````Python\n")
            file.write(str(job_stats["model"]["model_fn"]) + "\n")
            file.write("````\n")
        with open(
            os.path.join(job_stats_directory, "estimator.md"), "w"
        ) as file:
            file.write("## Train Hooks\n")
            file.write("````Python\n")
            file.write(str(job_stats["estimator"]["train_hooks"]) + "\n")
            file.write("````\n")
            file.write("## Eval Hooks\n")
            file.write("````Python\n")
            file.write(str(job_stats["estimator"]["eval_hooks"]) + "\n")
            file.write("````\n")
            file.write("## Train Fn\n")
            file.write("````Python\n")
            file.write(str(job_stats["estimator"]["train_fn"]) + "\n")
            file.write("````\n")
            file.write("## Eval Fn\n")
            file.write("````Python\n")
            file.write(str(job_stats["estimator"]["eval_fn"]) + "\n")
            file.write("````\n")
            file.write("## Train And Eval Fn\n")
            file.write("````Python\n")
            file.write(str(job_stats["estimator"]["train_eval_fn"]) + "\n")
            file.write("````\n")
        if len(job_stats["common"]) > 0:
            with open(
                os.path.join(job_stats_directory, "common.md"), "w"
            ) as file:
                file.write("## Model Common Functions\n")
                file.write("````Python\n")
                file.write(str(job_stats["common"]) + "\n")
                file.write("````\n")
