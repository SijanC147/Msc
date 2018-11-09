from os import listdir, makedirs
from os.path import join, isfile, exists
from shutil import rmtree
import comet_ml
from tsaplay.utils.io import start_tensorboard, restart_tf_serve_container
from tsaplay.constants import (
    EXPERIMENT_DATA_PATH,
    EXPORTS_DATA_PATH,
    RANDOM_SEED,
    SAVE_SUMMARY_STEPS,
    SAVE_CHECKPOINTS_STEPS,
    LOG_STEP_COUNT_STEPS,
    KEEP_CHECKPOINT_MAX,
)


class Experiment:
    def __init__(
        self,
        feature_provider,
        model,
        contd_tag=None,
        comet_api=None,
        run_config=None,
        job_dir=None,
    ):
        self.feature_provider = feature_provider
        self.model = model
        self.contd_tag = contd_tag
        self.experiment_dir = self._make_experiment_dir(job_dir)
        run_config = {
            "model_dir": join(self.experiment_dir),
            **(run_config or {}),
        }
        self.model.run_config = self.model.run_config.replace(
            **self.make_default_run_cofig(run_config)
        )
        if self.contd_tag and comet_api:
            self._setup_comet_ml_experiment(api_key=comet_api)

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
            model_export_dir = join(EXPORTS_DATA_PATH, export_model_name)
            if exists(model_export_dir) and overwrite:
                rmtree(model_export_dir)

            prev_exported_models = self.get_exported_models()
            self.model.export(
                directory=model_export_dir,
                feature_provider=self.feature_provider,
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
            m
            for m in listdir(EXPORTS_DATA_PATH)
            if not isfile(join(EXPORTS_DATA_PATH, m))
        ]

    def _make_experiment_dir(self, job_dir):
        data_root = job_dir or EXPERIMENT_DATA_PATH
        dir_parent = join(data_root, self.model.name)
        exp_dir_name = self.contd_tag or self.feature_provider.name
        exp_dir_name = exp_dir_name.replace(" ", "_")
        experiment_dir = join(dir_parent, exp_dir_name)
        if exists(experiment_dir) and self.contd_tag is None:
            i = 0
            while exists(experiment_dir):
                i += 1
                experiment_dir = join(dir_parent, exp_dir_name + "_" + str(i))
        makedirs(experiment_dir, exist_ok=True)
        return experiment_dir

    def _update_export_models_config(self):
        config_file_path = join(EXPORTS_DATA_PATH, "tfserve.conf")
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

        with open(config_file_path, "w") as config_file:
            config_file.write(config_file_str)

    @classmethod
    def make_default_run_cofig(cls, custom_config=None):
        custom_config = custom_config or {}

        # setting session config anything other than None in distributed
        # setting prevents device filters from being automatically set
        # job hangs indefinitely at final checkpoint
        session_config_keywords = ["session", "sess"]
        custom_sess_config = {
            "_".join(key.split("_")[1:]): value
            for key, value in custom_config.items()
            if key.split("_")[0] in session_config_keywords
        }
        default_session_config = {
            "allow_soft_placement": True,
            "log_device_placement": False,
        }
        default_session_config.update(custom_sess_config)

        custom_run_config = {
            key: value
            for key, value in custom_config.items()
            if key.split("_")[0] not in session_config_keywords
        }
        default_run_config = {
            "tf_random_seed": RANDOM_SEED,
            "save_summary_steps": SAVE_SUMMARY_STEPS,
            "save_checkpoints_steps": SAVE_CHECKPOINTS_STEPS,
            "log_step_count_steps": LOG_STEP_COUNT_STEPS,
            "keep_checkpoint_max": KEEP_CHECKPOINT_MAX,
            # "session_config": tf.ConfigProto(**default_session_config),
        }
        default_run_config.update(custom_run_config)

        return default_run_config

    def _setup_comet_ml_experiment(self, api_key):
        comet_key_file_path = join(self.experiment_dir, "_cometml.key")
        if exists(comet_key_file_path):
            with open(comet_key_file_path, "r") as comet_key_file:
                exp_key = comet_key_file.readline().strip()
        else:
            comet_experiment = comet_ml.Experiment(
                api_key=api_key, project_name=self.model.name, workspace="msc"
            )
            comet_experiment.set_name(self.contd_tag)
            exp_key = comet_experiment.get_key()
            with open(comet_key_file_path, "w+") as comet_key_file:
                comet_key_file.write(exp_key)
        self.model.attach_comet_ml_experiment(api_key, exp_key)
