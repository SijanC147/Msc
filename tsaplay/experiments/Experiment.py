import tensorflow as tfj
from shutil import rmtree
from os import getcwd, listdir
from os.path import join as _join, isfile, relpath, dirname, exists, abspath
from inspect import getfile
from tsaplay.utils._io import (
    start_tensorboard,
    write_stats_to_disk,
    restart_tf_serve_container,
)
import tsaplay.experiments._constants as EXPERIMENTS


class Experiment:
    def __init__(self, feature_provider, model, contd_tag=None):
        self.fp = feature_provider
        self.model = model

        if contd_tag is not None:
            self.contd_tag = contd_tag.replace(" ", "_").lower()
        else:
            self.contd_tag = None

        self._initialize_experiment_dir()

    def run(
        self,
        job,
        steps,
        early_stopping=False,
        hooks=[],
        debug=False,
        start_tb=False,
        tb_port=6006,
        debug_port=6064,
    ):
        if job == "train":
            stats = self.model.train(
                feature_provider=self.fp, steps=steps, hooks=hooks
            )
            write_stats_to_disk(
                job="train", stats=stats, path=self._experiment_dir
            )
        elif job == "eval":
            stats = self.model.evaluate(feature_provider=self.fp, hooks=hooks)
            write_stats_to_disk(
                job="eval", stats=stats, path=self._experiment_dir
            )
        elif job == "train+eval":
            train, test = self.model.train_and_eval(
                feature_provider=self.fp,
                steps=steps,
                early_stopping=early_stopping,
            )
            write_stats_to_disk(
                job="train", stats=train, path=self._experiment_dir
            )
            write_stats_to_disk(
                job="eval", stats=test, path=self._experiment_dir
            )

        if start_tb:
            start_tensorboard(
                model_dir=self.model.run_config.model_dir,
                port=tb_port,
                debug=debug,
                debug_port=debug_port,
            )

    def export_model(self, overwrite=False, restart_tfserve=False):
        if self.contd_tag is None:
            print("No continue tag defined, nothing to export!")
        else:
            export_model_name = self.model.name.lower() + "_" + self.contd_tag
            model_export_dir = _join(
                EXPERIMENTS.EXPORT_PATH, export_model_name
            )
            if exists(model_export_dir) and overwrite:
                rmtree(model_export_dir)

            prev_exported_models = self._list_exported_models()
            self.model.export(
                directory=model_export_dir,
                embedding_params=self.fp.embedding_params,
            )

            if prev_exported_models != self._list_exported_models():
                print("Updating tfserve.conf with new exported model info")
                self._update_export_models_config()
                if restart_tfserve:
                    print("Restarting tsaplay docker container")
                    logs = restart_tf_serve_container()
                    print(logs)

    def _initialize_experiment_dir(self):
        rel_model_path = _join(
            relpath(
                dirname(getfile(self.model.__class__)),
                _join(getcwd(), "tsaplay", "models"),
            ),
            self.model.name,
        )
        exp_dir_name = self.contd_tag or self.fp.name

        exp_dir = _join(EXPERIMENTS.DATA_PATH, rel_model_path, exp_dir_name)
        if exists(exp_dir) and self.contd_tag is None:
            i = 0
            while exists(exp_dir):
                i += 1
                exp_dir = _join(
                    EXPERIMENTS.DATA_PATH,
                    rel_model_path,
                    exp_dir_name + "_" + str(i),
                )
        summary_dir = _join(exp_dir, "tb_summary")
        self.model.run_config.replace(model_dir=summary_dir)
        self._experiment_dir = exp_dir

    def _update_export_models_config(self):
        config_file = _join(EXPERIMENTS.EXPORT_PATH, "tfserve.conf")
        names, base_paths = self._get_exported_models_names_and_paths(
            container_base="models"
        )
        config_file_str = "model_config_list: {\n"
        for name, path in zip(names, base_paths):
            config_file_str += (
                "    config: { \n"
                '        name: "' + name + '",\n'
                '        base_path: "' + path + '",\n'
                '        model_platform: "tensorflow"\n'
                "    }\n"
            )
        config_file_str += "}"

        with open(config_file, "w") as f:
            f.write(config_file_str)

    def _list_exported_models(self):
        exported_models = [
            m
            for m in listdir(EXPERIMENTS.EXPORT_PATH)
            if not isfile(_join(EXPERIMENTS.EXPORT_PATH, m))
        ]
        return exported_models

    def _get_exported_models_names_and_paths(self, container_base):
        models = self._list_exported_models()
        names = ["".join([m[0] for m in model.split("_")]) for model in models]

        for name in names:
            indices = [i for i, a in enumerate(names) if a == name]
            if len(indices) > 1:
                cnt = 1
                for index in indices[1:]:
                    names[index] = name + str(cnt)
                    cnt += 1

        paths = [("/" + container_base + "/" + model) for model in models]

        return names, paths
