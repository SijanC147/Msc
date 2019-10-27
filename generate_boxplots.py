# pylint: disable=too-many-locals,too-many-statements,wrong-import-position
import warnings
import argparse
from os import path, makedirs, getcwd, environ
import re
from datetime import datetime
import json
from itertools import groupby
from operator import itemgetter
import comet_ml
import numpy as np
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerBase
import matplotlib.text as mtext
import pandas as pd
import seaborn as sns


with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)  # noqa
    from tsaplay.utils.io import pickle_file, unpickle_file, args_to_dict

REPORTED = {
    "lstm": {
        # ! Scores from original paper
        "Tang et al. 2016 (Original)": {
            "dong": {"Micro-F1": 66.5, "Macro-F1": 64.7}
        },
        # ! Scores from LCR-ROT paper
        "Zheng et al. 2018": {
            "laptops": {"Micro-F1": 66.5},
            "restaurants": {"Micro-F1": 74.3},
            "dong": {"Micro-F1": 66.5},
        },
        # ! Scores from Reproduction paper
        "Moore et al. 2018": {"dong": {"Macro-F1": 60.69}},
    },
    "tdlstm": {
        # ! Scores from original paper
        "Tang et al. 2016 (Original)": {
            "dong": {"Micro-F1": 70.8, "Macro-F1": 69.0}
        },
        # ! Scores from RAM paper
        "Chen et al. 2017": {
            "laptops": {"Micro-F1": 71.83, "Macro-F1": 68.43},
            "restaurants": {"Micro-F1": 78.0, "Macro-F1": 66.73},
            "dong": {"Micro-F1": 66.62, "Macro-F1": 64.01},
        },
        # ! Scores from LCR-ROT paper
        "Zheng et al. 2018": {
            "laptops": {"Micro-F1": 68.1},
            "restaurants": {"Micro-F1": 75.6},
            "dong": {"Micro-F1": 70.8},
        },
        # ! Scores from Reproduction paper
        "Moore et al. 2018": {"dong": {"Macro-F1": 65.63}},
    },
    "tclstm": {
        # ! Scores from original paper
        "Tang et al. 2016 (Original)": {
            "dong": {"Micro-F1": 71.5, "Macro-F1": 69.5}
        },
        # ! Scores from Reproduction paper
        "Moore et al. 2018": {"dong": {"Macro-F1": 65.23}},
    },
    "memnet": {
        # ! Scores from original paper
        "Tang et al. 2016 (Original)": {
            "laptops": {"Micro-F1": 72.37},
            "restaurants": {"Micro-F1": 80.95},
        },
        # ! Scores from RAM
        "Chen et al. 2017": {
            "laptops": {"Micro-F1": 70.33, "Macro-F1": 64.09},
            "restaurants": {"Micro-F1": 78.16, "Macro-F1": 65.83},
            "dong": {"Micro-F1": 68.5, "Macro-F1": 66.91},
        },
        # ! Scores from LCR-ROT paper
        "Zheng et al. 2018": {
            "laptops": {"Micro-F1": 70.33},
            "restaurants": {"Micro-F1": 79.98},
            "dong": {"Micro-F1": 70.52},
        },
    },
    "ian": {
        # ! Scores from original paper
        "Dehong Ma et al. 2017 (Original)": {
            "laptops": {"Micro-F1": 74.49, "Macro-F1": 71.35},
            "restaurants": {"Micro-F1": 80.23, "Macro-F1": 70.8},
        },
        # ! Scores from LCR-ROT
        "Zheng et al. 2018": {
            "laptops": {"Micro-F1": 72.1},
            "restaurants": {"Micro-F1": 78.6},
        },
    },
    "ram": {
        # ! Scores from original paper
        "Chen et al. 2017 (Original)": {
            "laptops": {"Micro-F1": 74.49, "Macro-F1": 71.35},
            "restaurants": {"Micro-F1": 80.23, "Macro-F1": 70.8},
            "dong": {"Micro-F1": 69.36, "Macro-F1": 67.3},
        }
    },
    "lcrrot": {
        # ! Scores from original paper
        "Zheng et al. 2018 (Original)": {
            "restaurants": {"Micro-F1": 81.34},
            "laptops": {"Micro-F1": 75.24},
            "dong": {"Micro-F1": 72.69},
        }
    },
}

CMT_VALS_MAPPING = {
    "Hidden Units": {"cmt_key": "Hidden Units"},
    "LSTM Hidden Units": {"cmt_key": "Lstm Hidden Units"},
    "GRU Hidden Units": {"cmt_key": "Gru Hidden Units"},
    "LSTM Layers": {"cmt_key": "N Lstm Layers"},
    "L2 Weight": {"cmt_key": "L2 Weight"},
    "Dropout Rate": {"cmt_key": "Keep Prob"},
    "Train Embeddings": {
        "cmt_key": "Train Embeddings",
        "df_valfn": lambda v: "WE Trained"
        if v.lower() == "true"
        else "WE Not Trained",
        "default": "WE Trained",
        "incl_xaxis_label": False,
    },
    "Number of Hops": {"cmt_key": "N Hops", "xaxis_label": "# Hops"},
    "Learning Rate": {"cmt_key": "Learning Rate"},
    "Momentum": {"cmt_key": "Momentum"},
    "Optimizer": {
        "cmt_key": "Optimizer",
        "df_valfn": lambda v: {"GradientDescent": "SGD"}.get(
            v.replace("Optimizer", ""), v.replace("Optimizer", "")
        ),
    },
    "Kernel Initializer": {"cmt_key": "Initializer"},
    "Bias Initializer": {"cmt_key": "Bias Initializer"},
    "Batch Size": {"cmt_key": "Batch Size"},
    "OOV Initializer": {
        "cmt_key": "OOV Fn",
        "df_valfn": lambda v: str(v).capitalize(),
    },
    "OOV Threshold": {"cmt_key": "OOV Train"},
    "OOV Buckets": {"cmt_key": "OOV Buckets"},
    "EA Metric": {
        "cmt_key": "Early Stopping Metric",
        "df_valfn": lambda v: str(v).capitalize(),
        "use_xlabel": False,
    },
    "EA Patience": {"cmt_key": "Early Stopping Patience", "use_xlabel": False},
    "EA Min Epochs": {
        "cmt_key": "Early Stopping Minimum Iter",
        "use_xlabel": False,
    },
    "EA Max Epochs": {"cmt_key": "Epochs", "use_xlabel": False},
}

MODELS = {
    "ian": "IAN",
    "lcrrot": "LCR-Rot",
    "lstm": "LSTM",
    "tdlstm": "TD-LSTM",
    "tclstm": "TC-LSTM",
    "memnet": "MemNet",
    "ram": "RAM",
}

EMBEDDINGS = {
    "cc42": "GloVe CommonCrawl 42b (300d)",
    "cc840": "GloVe CommonCrawl 840b (300d)",
    "t200": "GloVe Twitter (200d)",
    "t100": "GloVe Twitter (100d)",
}


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models", nargs="+", choices=[*MODELS], required=False, default=None
    )
    parser.add_argument(
        "--params", "-p", nargs="*", required=False, default=None
    )
    return parser


def get_comet_api(api_key=None, **kwargs):
    api_key = api_key or environ.get("COMET_REST_API_KEY")
    api = comet_ml.papi.API(
        rest_api_key=api_key, cache=kwargs.get("cmt_cache", False)
    )
    if kwargs.get("cmt_cache", False):
        cached_endpoints = tuple(
            kwargs.get("cmt_cache_endpoints", ["experiments", "projects"])
        )
        api.do_cache(*cached_endpoints)
    return api


def get_metric_series(experiment, metric):
    return {
        v["epoch"]: float(v["metricValue"])
        for v in experiment.get_metrics()
        if v["metricName"] == metric
    }


def get_grouped_metric_series(project, workspace=None, metrics=None, **kwargs):
    api = get_comet_api(**kwargs)
    workspace = workspace or "reproduction"
    metrics = metrics or ["macro-f1", "micro-f1"]
    grouped_metrics = {}
    experiments = api.get_experiments(workspace, project_name=project)
    unique_experiment_names = np.unique([e.name for e in experiments]).tolist()
    for name in unique_experiment_names:
        grouped_metrics[name] = {
            "experiments": [e for e in experiments if e.name == name],
            **{
                m: [
                    get_metric_series(e, "eval_{}".format(m.lower()))
                    for e in experiments
                    if e.name == name
                ]
                for m in metrics
            },
        }

    return grouped_metrics, metrics


def save_plot(plot, model, plot_metric, **kwargs):
    save_format = kwargs.get("format", "pdf")
    fname = "{}-{}".format(
        plot_metric,
        kwargs.get("fname", datetime.now().strftime("%d%m%Y_%H%M%S")),
    )
    target_dir = path.join(
        kwargs.get("path", getcwd()),
        "figures",
        "reproduciton",
        MODELS.get(model),
    )
    if not path.exists(target_dir):
        makedirs(target_dir)
    target_path = path.join(target_dir, "{}.{}".format(fname, save_format))
    plot.savefig(target_path, format=save_format, bbox_inches="tight")


def is_currently_running(experiment):
    start = experiment.get_others_summary(other="train_START")
    end = experiment.get_others_summary(other="train_END")
    start = float(start[-1]) if len(start) > 0 else None
    end = float(end[-1]) if len(end) > 0 else None
    if end is None:
        return start is not None
    return start > end


def comet_to_df(workspace, models=None, metrics=None, **kwargs):
    models = models or [*MODELS]
    if models is not None and isinstance(models, str):
        models = [models]
    cached_data_path = path.join(
        path.dirname(__file__),
        "temp",
        f"{'comet_df' if models is None else '_'.join(sorted(models))}.pkl",
    )
    use_cached = kwargs.get("use_cached", False)
    if use_cached and path.exists(cached_data_path):
        return unpickle_file(cached_data_path)
    api = get_comet_api(**kwargs)
    workspace = workspace or "reproduction"
    metrics = metrics or ["Macro-F1", "Micro-F1"]
    datasets = kwargs.get("datasets", ["restaurants", "dong", "laptops"])
    embeddings = kwargs.get("embeddings", [*EMBEDDINGS])
    cols = (
        ["Workspace", "Experiment", "Model", "Dataset", "Embedding"]
        + metrics
        + ["Reported {}".format(m) for m in metrics]
        + [*CMT_VALS_MAPPING]
        + ["Total Vocabulary Size", "IV Size", "OOV Size"]
        + [
            "Train Vocabulary Size",
            "Train OOV",
            "Train OOV Embedded",
            "Train OOV Bucketed",
        ]
        + [
            "Test Vocabulary Size",
            "Test OOV",
            "Test OOV Bucketed",
            "Test Exclusive OOV",
        ]
    )
    data = []
    for prj in [p for p in api.get_projects(workspace) if p in models]:
        exp_groups, _ = get_grouped_metric_series(prj, metrics=metrics)
        for _, exp_group_runs in exp_groups.items():
            # ? (APIExperiment, ([{Epoch: Macro-F1}], [{Epoch: Micro-F1}]))
            for (exp, *run_metrics) in zip(*exp_group_runs.values()):
                if is_currently_running(exp):
                    continue
                exp_name_clean = exp.name  # pylint: disable=no-member
                model = exp.project_name  # pylint: disable=no-member
                exp_name_clean = exp_name_clean.replace(model, "")
                dataset = [ds for ds in datasets if ds in exp_name_clean][0]
                exp_name_clean = exp_name_clean.replace(dataset, "")
                embedding = [em for em in embeddings if em in exp_name_clean][
                    0
                ]
                exp_name_clean = exp_name_clean.replace(embedding, "")
                exp_name_clean = exp_name_clean.replace("-", " ").strip()
                cmt_params_others_summary = (
                    exp.get_others_summary() + exp.get_parameters_summary()
                )
                cmt_params_others_values = [
                    [
                        p["valueCurrent"]
                        if cmt_param_other.get("df_valfn") is None
                        else cmt_param_other["df_valfn"](p["valueCurrent"])
                        for p in cmt_params_others_summary
                        if re.match(
                            r"train_(AUTOPARAM: )?{}".format(
                                cmt_param_other["cmt_key"]
                            ),
                            p["name"],
                            re.IGNORECASE,
                        )
                    ]
                    for cmt_param_other in CMT_VALS_MAPPING.values()
                ]
                cmt_params_others_values = sum(
                    [
                        v[:1] if len(v) > 0 else [None]
                        for v in cmt_params_others_values
                    ],
                    [],
                )

                vocab_coverage = json.loads(
                    exp.get_asset(
                        asset_id=[
                            f["assetId"]
                            for f in exp.get_asset_list()
                            if f["fileName"] == "info.json"
                        ][0],
                        return_type="text",
                    )
                ).get("vocab_coverage", {})
                get_num = lambda p: int(
                    re.match(
                        r"(?P<number>[0-9]+)\s\((?P<percent>[^\%]+)\%\)", p
                    ).groupdict()["number"]
                )
                vocab_data = [
                    vocab_coverage["total"]["size"],
                    get_num(vocab_coverage["total"]["in_vocab"]),
                    get_num(vocab_coverage["total"]["out_of_vocab"]),
                    vocab_coverage["train"]["size"],
                    get_num(vocab_coverage["train"]["oov"]["total"]),
                    get_num(vocab_coverage["train"]["oov"]["embedded"]),
                    (
                        get_num(vocab_coverage["train"]["oov"]["bucketed"])
                        if vocab_coverage["train"]["oov"].get("bucketed")
                        else 0
                    ),
                    vocab_coverage["test"]["size"],
                    get_num(vocab_coverage["test"]["oov"]["total"]),
                    get_num(vocab_coverage["test"]["oov"]["bucketed"]),
                    (
                        get_num(vocab_coverage["test"]["oov"]["exclusive"])
                        if vocab_coverage["test"]["oov"].get("exclusive")
                        else 0
                    ),
                ]

                data += [
                    [
                        workspace,
                        exp_name_clean,
                        MODELS.get(model, model),
                        dataset.capitalize(),
                        EMBEDDINGS.get(embedding, embedding),
                    ]
                    + [
                        round(max(run_metric.values()) * 100, 2)
                        for run_metric in run_metrics
                    ]
                    + [
                        REPORTED.get(model, {})
                        .get(
                            (
                                [
                                    k
                                    for k in [*REPORTED.get(model, {})]
                                    if "original" in k.lower()
                                ]
                                or ["none"]
                            )[0],
                            {},
                        )
                        .get(dataset, {})
                        .get(m)
                        for m in metrics
                    ]
                    + cmt_params_others_values
                    + vocab_data
                ]
    data_frame = pd.DataFrame(data, columns=cols)
    for k, val in CMT_VALS_MAPPING.items():
        default_value = val.get("default")
        if default_value:
            data_frame[k] = data_frame[k].map(
                lambda el, d=default_value: el if el is not None else d
            )
    pickle_file(path=cached_data_path, data=data_frame)
    return data_frame


def group_citations(dataset, metric, model, reported=None):
    reported = reported or REPORTED
    cited_results = {
        citation: cited_datasets.get(dataset.lower(), {}).get(metric)
        for citation, cited_datasets in REPORTED[model].items()
        if (
            dataset.lower() in [*cited_datasets]
            and metric in [*cited_datasets.get(dataset.lower(), {})]
        )
    }
    cited_results = {
        "\n".join(list(map(itemgetter(0), v))): k
        for k, v in groupby(
            sorted(cited_results.items(), key=itemgetter(1)), itemgetter(1)
        )
    }
    return (
        {
            **{k: v for k, v in cited_results.items() if "(Original)" in k},
            **{
                k: v for k, v in cited_results.items() if "(Original)" not in k
            },
        }
        if cited_results
        else None
    )


# pylint: disable=abstract-method
class AnyObjectHandler(HandlerBase):
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        if isinstance(orig_handle, str):
            txt = mtext.Text(
                x=handlebox.xdescent,
                y=handlebox.ydescent,
                text=orig_handle,
                fontsize="large",
                linespacing=1,
            )
            handlebox.add_artist(txt)
            return txt


def draw_boxplot(models=None, **kwargs):
    df = comet_to_df("reproduction", models=models, **kwargs)
    plot_metrics = kwargs.get("plot_metrics", ["Macro-F1", "Micro-F1"])
    xticklabel_templates = kwargs.get("xticklabel_templates", dict())
    hparams = kwargs.get("hparams", [*CMT_VALS_MAPPING])
    models = models or [*MODELS]
    models_tiled = sum([[model] * len(plot_metrics) for model in models], [])
    xaxis_filters = kwargs.get("xaxis_filters")
    for (model, plot_metric) in zip(models_tiled, plot_metrics * len(models)):
        dfm = df[(df["Model"] == MODELS.get(model))]
        _xaxis_filters = (
            xaxis_filters.get(model)
            if xaxis_filters is not None
            and isinstance(xaxis_filters, dict)
            and model in [*xaxis_filters]
            else xaxis_filters
        )
        draw_boxplot.dfm = dfm
        draw_boxplot.flt = _xaxis_filters
        if _xaxis_filters:
            dfm = dfm.loc[
                (
                    dfm[list(_xaxis_filters)]
                    == pd.Series(
                        {
                            k: (str(v) if not isinstance(v, str) else v)
                            for k, v in _xaxis_filters.items()
                        }
                    )
                ).all(axis=1)
            ]
        if dfm.empty:
            warnings.warn("EMPTY DATAFRAME: {}({})".format(model, plot_metric))
            continue
        filter_by_ds = kwargs.get("datasets")
        if filter_by_ds:
            dfm = dfm[
                (
                    dfm["Dataset"].isin(
                        [filter_by_ds]
                        if isinstance(filter_by_ds, str)
                        else filter_by_ds
                    )
                )
            ]
        dss = dfm["Dataset"].unique().tolist()
        num_plots = len(dss)
        common_hparams = [
            (hparam, str(dfm[hparam].unique().tolist()[0]))
            for hparam in hparams
            if hparam in [*dfm]
            and len(dfm[hparam].unique().tolist()) == 1
            and dfm[hparam].unique().tolist()[0] is not None
        ]

        handles, labels, legends, colors = [], [], dict(), dict()
        fig, axes = plt.subplots(
            1, num_plots + 1, sharey=False, figsize=(16, 7)
        )
        fig.suptitle(
            "{}\n({})".format(model.upper(), plot_metric.replace("f1", "F1")),
            fontsize=14,
        )
        fig.subplots_adjust(top=0.85, wspace=(0.4 + (0.025 * num_plots)))

        xticklabel_factors = []
        xticklabel_template = (
            xticklabel_templates.get(model)
            if xticklabel_templates is not None
            and isinstance(xticklabel_templates, dict)
            and model in [*xticklabel_templates]
            else xticklabel_templates
        )
        if not xticklabel_template:
            xticklabel_factors = [
                f
                for f in [
                    C
                    for C, V in CMT_VALS_MAPPING.items()
                    if C not in [*map(itemgetter(0), common_hparams)]
                    and V.get("use_xlabel", True)
                    and C in [*dfm]
                ]
                if dfm[f].unique().tolist()[0] is not None
            ]
            if not xticklabel_factors and _xaxis_filters:
                xticklabel_factors = [*_xaxis_filters]
            xticklabel_template = ", ".join(
                [
                    "{{{}}}".format(factor)
                    if not CMT_VALS_MAPPING[factor].get("xtick_label")
                    else CMT_VALS_MAPPING[factor].get("xtick_label")
                    for factor in xticklabel_factors
                ]
            )
        elif xticklabel_template is not None:
            xticklabel_factors = re.findall(
                r"\{([^\}]*)?", xticklabel_template
            )

        for i, dataset_name in enumerate(dss):
            try:
                this_ax = axes[i % num_plots]
            except TypeError:
                this_ax = axes

            cited_results = group_citations(
                dataset=dataset_name, model=model, metric=plot_metric
            )
            if cited_results:
                cmap_name = kwargs.get("cited_cmap", "Accent")
                cmap_colors = matplotlib.cm.get_cmap(cmap_name).colors
                opp_ax = this_ax.twinx()
                opp_ax.tick_params(direction="out", length=0, color="none")
                opp_ax_vals = list(cited_results.values())
                opp_ax.set_yticks(opp_ax_vals)
                opp_ax.set_yticklabels(
                    ["{:.2f}%".format(val) for val in opp_ax_vals], va="center"
                )
                opp_ax.get_shared_y_axes().join(opp_ax, this_ax)
                loop_data = zip(
                    cited_results.items(),
                    opp_ax.get_yticklabels(),
                    cmap_colors[: len(cited_results)],
                )
                for (citation, value), lab, _color in loop_data:
                    lab.set_color(_color)
                    this_ax.axhline(y=value, color=_color, linestyle="--")
                    opp_ax.axhline(
                        y=value, color=_color, linestyle="--", label=citation
                    )
                legends[dataset_name] = opp_ax.get_legend_handles_labels()

            this_ax.set_title(dataset_name.capitalize())
            this_ax.grid(True, linestyle="dotted", which="both")
            dfmds = dfm[dfm["Dataset"] == dataset_name]
            box_palette = kwargs.get("box_palette", "muted")
            bp = sns.boxplot(
                y=plot_metric,
                x="Experiment",
                data=dfmds,
                palette=box_palette,
                hue="Embedding",
                width=0.4,
                ax=this_ax,
                linewidth=1.5,
                saturation=kwargs.get("box_sat", 0.75),
                showmeans=kwargs.get("showmeans", True),
                meanline=kwargs.get("meanline", False),
                meanprops=kwargs.get(
                    "meanprops",
                    {
                        "marker": "D",
                        "markeredgecolor": "black",
                        "markerfacecolor": kwargs.get(
                            "mean_facecolor", "ghostwhite"
                        ),
                    },
                ),
            )

            label_xaxis = (
                (len(xticklabel_template) > 0)
                if kwargs.get("label_xaxis") is None
                else kwargs.get("label_xaxis")
            )
            if label_xaxis:
                this_ax.set_xticklabels(
                    [
                        xticklabel_template.format(
                            **(
                                dfm[
                                    (dfm["Dataset"] == dataset_name)
                                    & (dfm["Experiment"] == lab.get_text())
                                ].to_dict("records")[0]
                            )
                        )
                        for lab in this_ax.get_xticklabels()
                    ]
                )
                xaxis_labels_height = -0.3
            else:
                this_ax.set_xticks([])
                xaxis_labels_height = -0.25
            this_ax.set_xlabel("")
            this_ax.set_ylabel("")

            for handle, label in zip(*bp.get_legend_handles_labels()):
                if label not in labels:
                    labels += [label]
                    handles += [handle]
                    if hasattr(handle, "get_facecolor"):
                        colors[label] = handle.get_facecolor()
            bp.get_legend().remove()

            cell_text = [
                sum(
                    list(
                        [
                            [],
                            [
                                "{:.2f}".format(
                                    round(
                                        dfmds[
                                            (dfmds["Embedding"] == em)
                                            & (dfmds["Experiment"] == exp)
                                        ][plot_metric].mean(),
                                        2,
                                    )
                                )
                                for em in dfmds[(dfmds["Experiment"] == exp)][
                                    "Embedding"
                                ]
                                .unique()
                                .tolist()
                            ],
                        ]
                        for exp in dfmds["Experiment"].unique().tolist()
                    )
                    + [[list()]],
                    [],
                ),
                sum(
                    list(
                        [
                            [],
                            [
                                "{:.2f}".format(
                                    dfmds[
                                        (dfmds["Embedding"] == em)
                                        & (dfmds["Experiment"] == exp)
                                    ][plot_metric].max()
                                )
                                for em in dfmds[(dfmds["Experiment"] == exp)][
                                    "Embedding"
                                ]
                                .unique()
                                .tolist()
                            ],
                        ]
                        for exp in dfmds["Experiment"].unique().tolist()
                    )
                    + [[list()]],
                    [],
                ),
                sum(
                    list(
                        [
                            [],
                            [
                                len(
                                    dfmds[
                                        (dfmds["Embedding"] == em)
                                        & (dfmds["Experiment"] == exp)
                                    ]
                                )
                                for em in dfmds[(dfmds["Experiment"] == exp)][
                                    "Embedding"
                                ]
                                .unique()
                                .tolist()
                            ],
                        ]
                        for exp in dfmds["Experiment"].unique().tolist()
                    )
                    + [[list()]],
                    [],
                ),
            ]

            cell_colors = [
                sum(
                    list(
                        [
                            ["none"],
                            [
                                colors[em][:-1]
                                + tuple([kwargs.get("table_opacity", 0.7)])
                                for em in dfmds[(dfmds["Experiment"] == exp)][
                                    "Embedding"
                                ]
                                .unique()
                                .tolist()
                            ],
                        ]
                        for exp in dfmds["Experiment"].unique().tolist()
                    )
                    + [[["none"]]],
                    [],
                )
            ] * len(cell_text)

            sep_width = 0.03
            num_sepcols = len([c for c in cell_text[0] if len(c) == 0])
            rem_width = 1 - (sep_width * num_sepcols)
            exp_widths = [
                rem_width / (len(cell_text[0]) - num_sepcols)
                if len(c) > 0
                else sep_width
                for c in cell_text[0]
            ]
            subcol_widths = [
                [w / max(len(exp), 1) for e in exp]
                if len(exp) > 0
                else [sep_width]
                for w, exp in zip(exp_widths, cell_text[0])
            ]

            table = bp.table(
                cellText=[
                    sum([c if len(c) > 0 else [""] for c in ct], [])
                    for ct in cell_text
                ],
                cellColours=[sum(cc, []) for cc in cell_colors],
                colWidths=sum(subcol_widths, []),
                cellLoc="center",
                loc="bottom",
                bbox=(0, xaxis_labels_height, 1, 0.2),
                rowLabels=["Mean(%)", "Max(%)", "N"],
            )
            table.auto_set_font_size(False)
            table.set_fontsize(12)

            for (y_pos, x_pos), cell in table.get_celld().items():
                if cell.get_text().get_text() == "":
                    cell.visible_edges = ""
                if x_pos == -1 or y_pos == 2:
                    cell.set_text_props(fontstyle="italic")
                    # pylint: disable=protected-access
                    cell._loc = "right" if x_pos == -1 else cell._loc

            this_ax.set(ylabel=("{}(%)".format(plot_metric) if i == 0 else ""))

        if label_xaxis:
            for ax in axes.flat[:-1]:
                ax.set(
                    xlabel=", ".join(
                        [
                            CMT_VALS_MAPPING[f].get("xaxis_label", f)
                            for f in xticklabel_factors
                            if CMT_VALS_MAPPING[f].get(
                                "incl_xaxis_label", True
                            )
                        ]
                    )
                )

        last_axis = axes[-1]
        last_axis.axis("off")
        legends = {**{"Embeddings": (handles, labels)}, **legends}
        all_handles = sum(
            [
                [group] + hnds + [matplotlib.patches.Patch(facecolor="none")]
                for group, (hnds, labs) in legends.items()
            ],
            [],
        )[:-1]
        all_labels = sum(
            [[""] + labs + [""] for group, (hnds, labs) in legends.items()], []
        )[:-1]

        last_axis.legend(
            all_handles,
            all_labels,
            loc="upper left",
            bbox_to_anchor=(-0.2, 0.5, 1, 0.5),
            labelspacing=0.7,
            borderpad=1,
            handler_map={object: AnyObjectHandler()},
        )

        if common_hparams:
            col_widths = [0.2 * num_plots, 1 - (0.2 * num_plots)]
            hparam_table = last_axis.table(
                cellText=[["Hyper-Parameter Configuration\n", ""]]
                + list(map(list, common_hparams)),
                colWidths=col_widths,
                cellLoc="left",
                bbox=(-0.2, xaxis_labels_height, 1, 0.3),
                edges="open",
            )
            hparam_table.auto_set_font_size(False)
            hparam_table.set_fontsize(7)
            for (y_pos, x_pos), cell in hparam_table.get_celld().items():
                if x_pos == 0 and y_pos == 0:
                    cell.set_text_props(fontsize="x-small")

        if kwargs.get("fname", False) or kwargs.get("format", False):
            save_plot(plt, model, plot_metric, **kwargs)
        else:
            plt.show()


def main():
    parser = argument_parser()
    args = parser.parse_args()
    params = args_to_dict(args.params)
    draw_boxplot(models=args.models, **params)


if __name__ == "__main__":
    main()
