import inspect
import json
from datetime import datetime
from functools import wraps
import tensorflow as tf
from tensorflow.estimator import ModeKeys  # noqa
from tsaplay.utils.tf import scaffold_init_fn_on_spec
from tsaplay.utils.io import temp_pngs
from tsaplay.utils.draw import plot_distributions, draw_venn
from tsaplay.utils.data import class_dist_stats
from tsaplay.hooks import LogProgressToComet


def cometml(model_fn):
    @wraps(model_fn)
    def wrapper(self, features, labels, mode, params):
        comet = self.comet_experiment
        if comet is not None and mode in [ModeKeys.TRAIN, ModeKeys.EVAL]:
            if mode == ModeKeys.TRAIN:
                run_config = self.run_config.__dict__
                comet_pretty_log(comet, self.aux_config, prefix="AUX")
                comet_pretty_log(comet, run_config, prefix="RUNCONFIG")
                comet_pretty_log(comet, params, hparams=True)
                comet.set_code(inspect.getsource(self.__class__))
                comet.set_filename(inspect.getfile(self.__class__))
                comet.set_epoch(0)

            spec = model_fn(self, features, labels, mode, params)

            if mode == ModeKeys.TRAIN:

                def export_graph_to_comet(sess):
                    comet.set_model_graph(sess.graph)

                spec = scaffold_init_fn_on_spec(spec, export_graph_to_comet)
                train_hooks = list(spec.training_hooks) or []
                train_hooks += [
                    LogProgressToComet(
                        mode=ModeKeys.TRAIN,
                        comet=comet,
                        epochs=params.get("epochs"),
                        epoch_steps=params["epoch_steps"],
                    )
                ]
                spec = spec._replace(training_hooks=train_hooks)
            elif mode == ModeKeys.EVAL:
                eval_hooks = list(spec.evaluation_hooks) or []
                eval_hooks += [
                    LogProgressToComet(
                        mode=ModeKeys.EVAL,
                        comet=comet,
                        epoch_steps=params["epoch_steps"],
                    )
                ]
                spec = spec._replace(evaluation_hooks=eval_hooks)

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
            _, initializer_type, _ = inspect.getmro(
                type(tf.initializers.identity())
            )
            if isinstance(value, initializer_type):
                initializer_name = type(value).__name__.replace("Random", "")
                initializer_args = {
                    arg: val
                    for arg, val in value.get_config().items()
                    if arg not in ["seed", "dtype"]
                }
                value = "{}{}".format(
                    initializer_name,
                    (
                        ""
                        if len(initializer_args) == 0
                        else "[{minval},{maxval}]".format(**initializer_args)
                        if len(initializer_args) == 2
                        and "minval" in initializer_args
                        and "maxval" in initializer_args
                        else "[{}]".format(
                            ",".join(
                                "{}={}".format(k, v)
                                for k, v in initializer_args.items()
                            )
                        )
                    ),
                )
            else:
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


def log_vocab_venn(comet_experiment, feature_provider):
    if comet_experiment is None:
        return
    _v = feature_provider.vocab_data
    threshold = _v["_threshold"]
    v_orig = _v["embedding"]["original"]
    v_extd = _v["embedding"]["extended"]
    v_train = _v["datasets"]["train"]["total"]
    v_train_oov = _v["datasets"]["train"]["oov"]["total"]
    v_over_t = _v["datasets"]["train"]["oov"]["embedded"]
    v_test = _v["datasets"]["test"]["total"]
    v_test_oov = _v["datasets"]["test"]["oov"]["total"]

    venn_images = {
        "Original Embedding": draw_venn(
            "Original Embedding", Embedding=v_orig, Train=v_train, Test=v_test
        ),
        "Extended Embedding": draw_venn(
            "Extended Embedding", Extended=v_extd, Train=v_train, Test=v_test
        ),
        "Out of Vocabulary": draw_venn(
            "Out of Vocabulary",
            **{
                "Train": v_train_oov,
                "Test": v_test_oov,
                "Over T(={})".format(threshold): v_over_t,
            }
        ),
    }

    for temp_png in temp_pngs(list(venn_images.values()), [*venn_images]):
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
