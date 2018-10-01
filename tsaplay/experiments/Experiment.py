import tensorflow as tf
from shutil import rmtree
from os import getcwd, listdir
from os.path import join as join, isfile, relpath, dirname, exists, abspath
from inspect import getfile
from tsaplay.utils.io import start_tensorboard, restart_tf_serve_container

tf.logging.set_verbosity(tf.logging.INFO)

DATA_PATH = join(getcwd(), "tsaplay", "experiments", "data")
EXPORT_PATH = join(getcwd(), "export")


class Experiment:
    def __init__(self, feature_provider, model, contd_tag=None):
        self.fp = feature_provider
        self.model = model

        if contd_tag is not None:
            self.contd_tag = contd_tag.replace(" ", "_").lower()
        else:
            self.contd_tag = None

        self._initialize_experiment_dir()

    def run(self, job, steps):
        if job == "train":
            self.model.train(feature_provider=self.fp, steps=steps)
        elif job == "eval":
            self.model.evaluate(feature_provider=self.fp)
        elif job == "train+eval":
            self.model.train_and_eval(feature_provider=self.fp, steps=steps)

    def launch_tensorboard(self, tb_port=6006, debug=False, debug_port=6064):
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
            model_export_dir = join(EXPORT_PATH, export_model_name)
            if exists(model_export_dir) and overwrite:
                rmtree(model_export_dir)

            prev_exported_models = self._exported_models_list()
            self.model.export(
                directory=model_export_dir,
                embedding_params=self.fp.embedding_params,
            )

            if prev_exported_models != self._exported_models_list():
                print("Updating tfserve.conf with new exported model info")
                self._update_export_models_config()
                if restart_tfserve:
                    print("Restarting tsaplay docker container")
                    logs = restart_tf_serve_container()
                    print(logs)

    def _initialize_experiment_dir(self):
        exp_dir_name = self.contd_tag or self.fp.name

        experiment_dir = join(DATA_PATH, self.model.name, exp_dir_name)
        if exists(experiment_dir) and self.contd_tag is None:
            i = 0
            while exists(experiment_dir):
                i += 1
                experiment_dir = join(
                    DATA_PATH, self.model.name, exp_dir_name + "_" + str(i)
                )
        summary_dir = join(experiment_dir, "tb_summary")
        self.model.run_config = self.model.run_config.replace(
            model_dir=summary_dir
        )
        self._experiment_dir = experiment_dir

    def _update_export_models_config(self):
        config_file = join(EXPORT_PATH, "tfserve.conf")
        config_file_str = "model_config_list: {\n"
        for model in self._exported_models_list():
            config_file_str += (
                "    config: { \n"
                '        name: "' + model + '",\n'
                '        base_path: "/models/' + model + '/",\n'
                '        model_platform: "tensorflow"\n'
                "    }\n"
            )
        config_file_str += "}"

        with open(config_file, "w") as f:
            f.write(config_file_str)

    def _exported_models_list(self):
        exported_models = [
            m for m in listdir(EXPORT_PATH) if not isfile(join(EXPORT_PATH, m))
        ]
        return exported_models
