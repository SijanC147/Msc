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
        self.embedding = (
            GloVe(alias="twitter", version="debug") if debug else embedding
        )
        self.dataset = Dong2014() if debug else dataset
        self.dataset.set_embedding(self.embedding)
        self.model = model
        self.experiment_directory = self.init_experiment_directory(
            model=self.model,
            dataset=self.dataset,
            custom_tag=custom_tag,
            continue_training=continue_training,
        )
        summary_dir = os.path.join(self.experiment_directory, "tb_summary")

        self.run_config = (
            tf.estimator.RunConfig(model_dir=summary_dir)
            if run_config is None
            else run_config
        )

        if self.run_config.model_dir is None:
            self.run_config = run_config.replace(model_dir=summary_dir)

        if seed is not None:
            self.run_config = self.run_config.replace(tf_random_seed=seed)

        self.dataset.set_embedding(self.embedding)

        # self.model.embedding = self.embedding
        # self.model.dataset = self.dataset
        self.model.run_config = self.run_config
        # self.model.initialize_internal_defaults()

    def init_experiment_directory(
        self, model, dataset, custom_tag, continue_training
    ):
        all_experiments_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "data"
        )
        relative_model_path = os.path.join(
            os.path.relpath(
                os.path.dirname(inspect.getfile(model.__class__)),
                os.path.join(os.getcwd(), "models"),
            ),
            model.__class__.__name__,
        )
        if len(custom_tag) > 0:
            experiment_folder_name = "_".join(
                [
                    dataset.__class__.__name__,
                    dataset.embedding.__class__.__name__,
                    dataset.embedding.alias,
                    dataset.embedding.version,
                    custom_tag.replace(" ", "_"),
                ]
            )
        else:
            experiment_folder_name = "_".join(
                [
                    dataset.__class__.__name__,
                    dataset.embedding.__class__.__name__,
                    dataset.embedding.alias,
                    dataset.embedding.version,
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
        # self.model.create_estimator()
        if job == "train":
            stats = self.model.train(
                dataset=self.dataset, steps=steps, debug=debug
            )
            self._write_stats_to_experiment_dir(job="train", stats=stats)
        elif job == "eval":
            stats = self.model.evaluate(dataset=self.dataset, debug=debug)
            self._write_stats_to_experiment_dir(job="eval", stats=stats)
        elif job == "train+eval":
            train_stats, eval_stats = self.model.train_and_evaluate(
                steps=steps, dataset=self.dataset
            )
            self._write_stats_to_experiment_dir(job="train", stats=train_stats)
            self._write_stats_to_experiment_dir(job="eval", stats=eval_stats)

        if start_tensorboard:
            start_tensorboard(model_dir=self.run_config.model_dir, debug=debug)

    def _write_stats_to_experiment_dir(self, job, stats):
        job_stats_directory = os.path.join(self.experiment_directory, job)
        os.makedirs(job_stats_directory, exist_ok=True)
        with open(
            os.path.join(job_stats_directory, "dataset.json"), "w"
        ) as file:
            file.write(json.dumps(stats["dataset"]))
        with open(os.path.join(job_stats_directory, "job.json"), "w") as file:
            file.write(
                json.dumps(
                    {"duration": stats["duration"], "steps": stats["steps"]}
                )
            )
        with open(os.path.join(job_stats_directory, "model.md"), "w") as file:
            file.write("## Model Params\n")
            file.write("````Python\n")
            file.write(str(stats["model"]["params"]) + "\n")
            file.write("````\n")
            file.write("## Train Input Fn\n")
            file.write("````Python\n")
            file.write(str(stats["model"]["train_input_fn"]) + "\n")
            file.write("````\n")
            file.write("## Eval Input Fn\n")
            file.write("````Python\n")
            file.write(str(stats["model"]["eval_input_fn"]) + "\n")
            file.write("````\n")
            file.write("## Model Fn\n")
            file.write("````Python\n")
            file.write(str(stats["model"]["model_fn"]) + "\n")
            file.write("````\n")
        with open(
            os.path.join(job_stats_directory, "estimator.md"), "w"
        ) as file:
            file.write("## Train Hooks\n")
            file.write("````Python\n")
            file.write(str(stats["estimator"]["train_hooks"]) + "\n")
            file.write("````\n")
            file.write("## Eval Hooks\n")
            file.write("````Python\n")
            file.write(str(stats["estimator"]["eval_hooks"]) + "\n")
            file.write("````\n")
            file.write("## Train Fn\n")
            file.write("````Python\n")
            file.write(str(stats["estimator"]["train_fn"]) + "\n")
            file.write("````\n")
            file.write("## Eval Fn\n")
            file.write("````Python\n")
            file.write(str(stats["estimator"]["eval_fn"]) + "\n")
            file.write("````\n")
            file.write("## Train And Eval Fn\n")
            file.write("````Python\n")
            file.write(str(stats["estimator"]["train_eval_fn"]) + "\n")
            file.write("````\n")
        if len(stats["common"]) > 0:
            with open(
                os.path.join(job_stats_directory, "common.md"), "w"
            ) as file:
                file.write("## Model Common Functions\n")
                file.write("````Python\n")
                file.write(str(stats["common"]) + "\n")
                file.write("````\n")
