from os import getcwd, listdir, environ, makedirs
from os.path import join, isfile, exists
from shutil import rmtree

import comet_ml
import tensorflow as tf
from tsaplay.utils.io import start_tensorboard, restart_tf_serve_container

tf.logging.set_verbosity(tf.logging.INFO)

DATA_PATH = join(getcwd(), "tsaplay", "experiments", "data")
EXPORT_PATH = join(getcwd(), "export")


class Experiment:
    def __init__(self, feature_provider, model, contd_tag=None, config=None):
        self.feature_provider = feature_provider
        self.model = model
        self.contd_tag = contd_tag
        self._initialize_experiment_dir()
        self._initialize_model_run_config(config or {})
        if self.contd_tag is not None:
            self._setup_comet_ml_experiment()

    def run(self, job, steps):
        if job == "train":
            self.model.train(
                feature_provider=self.feature_provider, steps=steps
            )
        elif job == "eval":
            self.model.evaluate(feature_provider=self.feature_provider)
        elif job == "train+eval":
            self.model.train_and_eval(
                feature_provider=self.feature_provider, steps=steps
            )

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

            prev_exported_models = self.get_exported_models()
            self.model.export(
                directory=model_export_dir,
                embedding_params=self.feature_provider.embedding_params,
            )

            if prev_exported_models != self.get_exported_models():
                print("Updating tfserve.conf with new exported model info")
                self._update_export_models_config()
                if restart_tfserve:
                    print("Restarting tsaplay docker container")
                    logs = restart_tf_serve_container()
                    print(logs)

    @classmethod
    def get_exported_models(cls):
        return [
            m for m in listdir(EXPORT_PATH) if not isfile(join(EXPORT_PATH, m))
        ]

    def _initialize_experiment_dir(self):
        dir_parent = join(DATA_PATH, self.model.name)
        exp_dir_name = self.contd_tag or self.feature_provider.name
        experiment_dir = join(dir_parent, exp_dir_name)
        if exists(experiment_dir) and self.contd_tag is None:
            i = 0
            while exists(experiment_dir):
                i += 1
                experiment_dir = join(dir_parent, exp_dir_name + "_" + str(i))
        makedirs(experiment_dir, exist_ok=True)
        self._experiment_dir = experiment_dir

    def _update_export_models_config(self):
        config_file = join(EXPORT_PATH, "tfserve.conf")
        config_file_str = "model_config_list: {\n"
        for model in self.get_exported_models():
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

    def _initialize_model_run_config(self, config_dict):
        config_dict.update(
            {
                "model_dir": join(self._experiment_dir, "tb_summary"),
                "save_checkpoints_steps": 500,
                "save_summary_steps": 10,
                "log_step_count_steps": 10,
            }
        )
        self.model.params.update(config_dict)
        self.model.run_config = tf.estimator.RunConfig(**config_dict)

    def _setup_comet_ml_experiment(self):
        api_key = environ.get("COMET_ML_API_KEY")
        if api_key is not None:
            comet_key_file = join(self._experiment_dir, "_cometml.key")
            if exists(comet_key_file):
                with open(comet_key_file, "r") as f:
                    exp_key = f.readline().strip()
            else:
                comet_experiment = comet_ml.Experiment(
                    api_key=api_key,
                    project_name=self.model.name,
                    workspace="msc",
                )
                comet_experiment.set_name(self.contd_tag)
                exp_key = comet_experiment.get_key()
                with open(comet_key_file, "w+") as f:
                    f.write(exp_key)
            self.model.attach_comet_ml_experiment(api_key, exp_key)
