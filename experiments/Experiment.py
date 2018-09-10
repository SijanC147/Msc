import tensorflow as tf
from os import getcwd
from os.path import join as _join, relpath, dirname, exists, abspath
from inspect import getfile
from utils import start_tensorboard, write_stats_to_disk


class Experiment:
    def __init__(
        self, dataset, embedding, model, run_config=None, custom_tag=""
    ):
        self.embedding = embedding
        self.dataset = dataset
        self.dataset.embedding = self.embedding
        self.model = model
        self.exp_dir = self._init_exp_dir(
            model=self.model, dataset=self.dataset, custom_tag=custom_tag
        )
        summary_dir = _join(self.exp_dir, "tb_summary")

        self.run_config = (
            tf.estimator.RunConfig(model_dir=summary_dir)
            if run_config is None
            else run_config
        )

        if self.run_config.model_dir is None:
            self.run_config = run_config.replace(model_dir=summary_dir)

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

    def _init_exp_dir(self, model, dataset, custom_tag):
        all_exps_path = _join(dirname(abspath(__file__)), "data")
        rel_model_path = _join(
            relpath(
                dirname(getfile(model.__class__)), _join(getcwd(), "models")
            ),
            model.__class__.__name__,
        )
        exp_dir_name = "_".join([dataset.name, dataset.embedding.version])
        if len(custom_tag) > 0:
            exp_dir_name += "_" + custom_tag.replace(" ", "_")

        exp_dir = _join(all_exps_path, rel_model_path, exp_dir_name)
        if exists(exp_dir) and not len(custom_tag) > 0:
            i = 0
            while exists(exp_dir):
                i += 1
                exp_dir = _join(
                    all_exps_path, rel_model_path, exp_dir_name + "_" + str(i)
                )
        return exp_dir
