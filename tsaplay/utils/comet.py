import inspect
import json
from math import ceil
from datetime import datetime
from functools import wraps
import tensorflow as tf
from tensorflow.estimator import ModeKeys  # pylint: disable=E0401
from tsaplay.utils.tf import scaffold_init_fn_on_spec
from tsaplay.utils.io import temp_pngs
from tsaplay.utils.draw import plot_distributions
from tsaplay.utils.data import class_dist_stats


def cometml(model_fn):
    @wraps(model_fn)
    def wrapper(self, features, labels, mode, params):
        comet = self.comet_experiment
        if comet is not None and mode in [ModeKeys.TRAIN, ModeKeys.EVAL]:
            run_config = self.run_config.__dict__
            comet_pretty_log(comet, self.aux_config, prefix="AUX")
            comet_pretty_log(comet, run_config, prefix="RUNCONFIG")
            comet_pretty_log(comet, params, hparams=True)
            comet.set_code(inspect.getsource(self.__class__))
            comet.set_filename(inspect.getfile(self.__class__))
            if mode == ModeKeys.TRAIN:
                global_step = tf.train.get_global_step()
                comet.set_step(global_step)
                if (
                    global_step == 1
                    or global_step % params["epoch_steps"] == 0
                ):
                    comet.log_current_epoch(
                        ceil(global_step / params["epoch_steps"])
                    )

            spec = model_fn(self, features, labels, mode, params)

            if mode == ModeKeys.TRAIN:

                def export_graph_to_comet(sess):
                    comet.set_model_graph(sess.graph)

                spec = scaffold_init_fn_on_spec(spec, export_graph_to_comet)

                if (
                    global_step == 1
                    or global_step % params["epoch_steps"] == 0
                ):
                    comet.log_epoch_end(
                        ceil(global_step / params["epoch_steps"])
                    )

            return spec
        return model_fn(self, features, labels, mode, params)

    return wrapper


def comet_pretty_log(comet_experiment, data_dict, prefix=None, hparams=False):
    for key, value in data_dict.items():
        if hparams and key[0] == "_" and prefix is None:
            prefix = "autoparam"
        log_as_param = hparams and key[0] != "_"
        key = key.replace("-", "_")
        key = key.split("_")
        key = " ".join(map(str.capitalize, key)).strip()
        if prefix:
            key = prefix.upper() + ": " + key
        try:
            json.dumps(value)
        except TypeError:
            value = str(value)
        if log_as_param:
            comet_experiment.log_parameter(key, value)
        else:
            comet_experiment.log_other(key, value)


def log_dist_data(comet_experiment, feature_provider, modes):
    if comet_experiment is None:
        return
    stats = {
        ds.name: class_dist_stats(
            ds.class_labels, train=ds.train_dict, test=ds.test_dict
        )
        for ds in feature_provider.datasets
    }
    dist_images = [plot_distributions(stats, mode) for mode in modes]
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    dist_image_names = [mode + "_distribution_" + timestamp for mode in modes]
    for temp_png in temp_pngs(dist_images, dist_image_names):
        comet_experiment.log_image(temp_png)


def log_features_asset_data(comet_experiment, feature_provider):
    if comet_experiment is None:
        return
    comet_experiment.log_asset_folder(feature_provider.gen_dir)
    filter_info = feature_provider.embedding.filter_info
    if filter_info:
        comet_experiment.log_other("Filter Hash", filter_info["hash"])
        comet_experiment.log_asset_data(
            json.dumps(filter_info["details"], indent=4),
            file_name="embedding_filter_details.json",
        )
        report = filter_info.get("report")
        if report:
            header = "".join(
                ["<th>{}</th>".format(heading) for heading in report[0]]
            )
            data = report[1:]
            data = "".join(
                [
                    "<tr>{}</tr>".format(
                        "".join(["<td>{}</td>".format(value) for value in row])
                    )
                    for row in data
                ]
            )
            table = "<table><tr>{0}</tr>{1}</table>".format(header, data)
            comet_experiment.log_html(table)
