import tensorflow as tf
from os import getcwd
from os.path import join as _join, relpath, dirname, exists, abspath
from inspect import getfile
from tsaplay.utils._io import start_tensorboard, write_stats_to_disk


class Experiment:
    def __init__(
        self, dataset, model, embedding=None, run_config=None, contd_tag=None
    ):
        self.dataset = dataset
        self.model = model
        self.model_name = model.__class__.__name__
        if embedding is not None:
            self.embedding = embedding
            self.dataset.embedding = self.embedding
        else:
            if self.dataset.embedding is None:
                raise ValueError(
                    "No embedding found in experiment or dataset."
                )
            else:
                self.embedding = self.dataset.embedding
        self.contd_tag = contd_tag
        self.exp_dir = self._init_exp_dir(
            model=self.model, dataset=self.dataset, contd_tag=self.contd_tag
        )
        if run_config is None:
            run_config = self.model.run_config
        self.model.run_config = self._init_model_dir(
            exp_dir=self.exp_dir, run_config=run_config
        )

    def run(
        self,
        job,
        steps,
        dist=None,
        hooks=[],
        debug=False,
        start_tb=False,
        tb_port=6006,
        debug_port=6064,
    ):
        if job == "train":
            stats = self.model.train(
                dataset=self.dataset,
                steps=steps,
                distribution=dist,
                hooks=hooks,
            )
            write_stats_to_disk(job="train", stats=stats, path=self.exp_dir)
        elif job == "eval":
            stats = self.model.evaluate(
                dataset=self.dataset, distribution=dist, hooks=hooks
            )
            write_stats_to_disk(job="eval", stats=stats, path=self.exp_dir)
        elif job == "train+eval":
            train, test = self.model.train_and_eval(
                dataset=self.dataset, steps=steps
            )
            write_stats_to_disk(job="train", stats=train, path=self.exp_dir)
            write_stats_to_disk(job="eval", stats=test, path=self.exp_dir)

        if start_tb:
            start_tensorboard(
                model_dir=self.model.run_config.model_dir,
                port=tb_port,
                debug=debug,
                debug_port=debug_port,
            )

    def export_model(self):
        if self.contd_tag is None:
            print("No continue tag defined, nothing to export!")
        else:
            export_model_name = "_".join(
                [self.model_name.lower(), self.contd_tag]
            )
            export_dir = _join(getcwd(), "export", export_model_name)
            self.model.export(directory=export_dir)
        return

    def _init_exp_dir(self, model, dataset, contd_tag):
        all_exps_path = _join(dirname(abspath(__file__)), "data")
        rel_model_path = _join(
            relpath(
                dirname(getfile(model.__class__)),
                _join(getcwd(), "tsaplay", "models"),
            ),
            model.__class__.__name__,
        )
        if contd_tag is not None:
            exp_dir_name = contd_tag.replace(" ", "_")
        else:
            exp_dir_name = "_".join([dataset.name, dataset.embedding.version])

        exp_dir = _join(all_exps_path, rel_model_path, exp_dir_name)
        if exists(exp_dir) and contd_tag is None:
            i = 0
            while exists(exp_dir):
                i += 1
                exp_dir = _join(
                    all_exps_path, rel_model_path, exp_dir_name + "_" + str(i)
                )
        return exp_dir

    def _init_model_dir(self, exp_dir, run_config):
        summary_dir = _join(exp_dir, "tb_summary")
        if run_config.model_dir is None:
            return run_config.replace(model_dir=summary_dir)
        else:
            return run_config
