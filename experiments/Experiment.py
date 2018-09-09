import os
import time
import inspect
import json
import tensorflow as tf
from utils import start_tensorboard, write_stats_to_disk
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
        self.exp_dir = self._init_exp_dir(
            model=self.model,
            dataset=self.dataset,
            custom_tag=custom_tag,
            continue_training=continue_training,
        )
        summary_dir = os.path.join(self.exp_dir, "tb_summary")

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

        self.model.run_config = self.run_config

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
            stats = self.model.train(
                dataset=self.dataset, steps=steps, debug=debug
            )
            write_stats_to_disk(job="train", stats=stats, path=self.exp_dir)
        elif job == "eval":
            stats = self.model.evaluate(dataset=self.dataset, debug=debug)
            write_stats_to_disk(job="eval", stats=stats, path=self.exp_dir)
        elif job == "train+eval":
            train_stats, eval_stats = self.model.train_and_evaluate(
                dataset=self.dataset, steps=steps
            )
            write_stats_to_disk(
                job="train", stats=train_stats, path=self.exp_dir
            )
            write_stats_to_disk(
                job="eval", stats=eval_stats, path=self.exp_dir
            )

        if start_tensorboard:
            start_tensorboard(model_dir=self.run_config.model_dir, debug=debug)

    def _init_exp_dir(self, model, dataset, custom_tag, continue_training):
        all_exps_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "data"
        )
        rel_model_path = os.path.join(
            os.path.relpath(
                os.path.dirname(inspect.getfile(model.__class__)),
                os.path.join(os.getcwd(), "models"),
            ),
            model.__class__.__name__,
        )
        exp_dir_name = "_".join(
            [
                dataset.__class__.__name__,
                dataset.embedding.__class__.__name__,
                dataset.embedding.alias,
                dataset.embedding.version,
            ]
        )
        if len(custom_tag) > 0:
            exp_dir_name += "_" + custom_tag.replace(" ", "_")

        exp_dir = os.path.join(all_exps_path, rel_model_path, exp_dir_name)
        if os.path.exists(exp_dir) and not (continue_training):
            i = 0
            while os.path.exists(exp_dir):
                i += 1
                exp_dir = os.path.join(
                    all_exps_path, rel_model_path, exp_dir_name + "_" + str(i)
                )
        return exp_dir
