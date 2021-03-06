from os import listdir, makedirs, sys
from os.path import join, isfile, exists
from shutil import rmtree
import re
import comet_ml
import tensorflow as tf
from tsaplay.utils.tf import checkpoints_state_data
from tsaplay.utils.io import (
    start_tensorboard,
    restart_tf_serve_container,
    extract_config_subset,
    cprnt,
)
from tsaplay.constants import (
    EXPERIMENT_DATA_PATH,
    EXPORTS_DATA_PATH,
    TF_RANDOM_SEED,
    SAVE_SUMMARY_STEPS,
    SAVE_CHECKPOINTS_STEPS,
    LOG_STEP_COUNT_STEPS,
    KEEP_CHECKPOINT_MAX,
    DEFAULT_COMET_WORKSPACE,
)


class Experiment:
    def __init__(
        self,
        feature_provider,
        model,
        contd_tag=None,
        comet_api=None,
        comet_workspace=None,
        run_config=None,
        job_dir=None,
    ):
        self.feature_provider = feature_provider
        self.model = model
        self.contd_tag = contd_tag
        self.model.params.update({"contd_tag": self.contd_tag})
        self.experiment_dir = self._make_experiment_dir(job_dir)
        self.model.run_config = self.model.run_config.replace(
            **self.make_default_run_cofig(run_config)
        )
        if self.contd_tag and comet_api:
            self._setup_comet_ml_experiment(
                api_key=comet_api, workspace=comet_workspace
            )

    def run(self, job, **kwargs):
        try:
            debug_port = int(self.model.aux_config.get("debug"))
            self.launch_tensorboard(debug=True, debug_port=debug_port)
        except (TypeError, ValueError):
            pass

        if job == "train":
            self.model.train(feature_provider=self.feature_provider, **kwargs)
        elif job == "eval":
            self.model.evaluate(feature_provider=self.feature_provider)
        elif job == "train+eval":
            self.model.train_and_eval(
                feature_provider=self.feature_provider, **kwargs
            )

    def launch_tensorboard(self, tb_port=6006, debug=False, debug_port=6064):
        start_tensorboard(
            model_dir=self.model.run_config.model_dir,
            port=tb_port,
            debug=debug,
            debug_port=debug_port,
            sub=True,
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
                cprnt(
                    info="Updating tfserve.conf with new exported model info"
                )
                self._update_export_models_config()
                if restart_tfserve:
                    cprnt(info="Restarting tsaplay docker container")
                    logs = restart_tf_serve_container()
                    cprnt(info=logs)

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
        chkpt = 0
        if exists(experiment_dir):
            if self.contd_tag is None:
                i = 0
                while exists(experiment_dir):
                    i += 1
                    experiment_dir = join(
                        dir_parent, exp_dir_name + "_" + str(i)
                    )
            elif self.contd_tag is not None:
                chkpt = checkpoints_state_data(experiment_dir).get("step")
        self.model.aux_config["chkpt"] = chkpt
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

    def make_default_run_cofig(self, custom_config=None):
        custom_config = custom_config or {}

        # setting session config anything other than None in distributed
        # setting prevents device filters from being automatically set
        # job hangs indefinitely at final checkpoint
        session_conf_keywords = ["session", "sess"]
        custom_sess_config = extract_config_subset(
            config_objs=[custom_config], keywords=session_conf_keywords
        )
        default_session_config = {
            "allow_soft_placement": True,
            "log_device_placement": False,
        }
        default_session_config.update(custom_sess_config)

        custom_run_config = {
            key: value
            for key, value in custom_config.items()
            if key.split("_")[0] not in session_conf_keywords
        }
        if "save_checkpoints_steps" in [*custom_run_config]:
            self.model.aux_config["checkpoints_freq"] = self.model.aux_config[
                "early_stopping_freq"
            ] = custom_run_config.pop("save_checkpoints_steps")
        elif "save_checkpoints_secs" in [*custom_run_config]:
            self.model.aux_config["checkpoints_secs"] = self.model.aux_config[
                "early_stopping_secs"
            ] = custom_run_config.pop("save_checkpoints_secs")
        if "save_summary_steps" in [*custom_run_config]:
            self.model.aux_config["summaries_freq"] = custom_run_config.pop(
                "save_summary_steps"
            )

        default_run_config = {
            # "session_config": tf.ConfigProto(**default_session_config),
            "model_dir": self.experiment_dir,
            "tf_random_seed": TF_RANDOM_SEED,
            "log_step_count_steps": LOG_STEP_COUNT_STEPS,
            "keep_checkpoint_max": KEEP_CHECKPOINT_MAX,
            # * these settings are delegated to the respective addons
            "save_checkpoints_steps": None,
            "save_checkpoints_secs": None,
            "save_summary_steps": None,
        }
        default_run_config.update(custom_run_config)
        if default_run_config["tf_random_seed"] is not None:
            cprnt(
                tf=True,
                warn="TF Random seed is set to {}".format(
                    default_run_config.get("tf_random_seed")
                ),
            )
        if default_run_config.get("keep_checkpoint_max") < 2:
            default_run_config["keep_checkpoint_max"] = 2
            cprnt(
                tf=True,
                warn=(
                    "TSAPLAY requires keep_checkpoint_max >=2"
                    + " got {}, using 2"
                ).format(default_run_config.get("keep_checkpoint_max")),
            )

        return default_run_config

    def _setup_comet_ml_experiment(self, api_key, workspace=None):
        comet_key_file_path = join(self.experiment_dir, "_cometml.key")
        if exists(comet_key_file_path):
            with open(comet_key_file_path, "r") as comet_key_file:
                exp_key = comet_key_file.readline().strip()
        else:
            comet_experiment = comet_ml.Experiment(
                api_key=api_key,
                project_name=self.model.name,
                workspace=(workspace or DEFAULT_COMET_WORKSPACE),
                log_env_details=False,
            )
            comet_experiment.set_name(self.contd_tag)
            exp_key = comet_experiment.get_key()
            with open(comet_key_file_path, "w+") as comet_key_file:
                comet_key_file.write(exp_key)
        self.model.attach_comet_ml_experiment(api_key, exp_key)
