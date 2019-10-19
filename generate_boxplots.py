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
import pandas as pd
import seaborn as sns

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)  # noqa
    from tsaplay.utils.io import pickle_file, unpickle_file

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
    },
    "tclstm": {
        # ! Scores from original paper
        "Tang et al. 2016 (Original)": {
            "dong": {"Micro-F1": 71.5, "Macro-F1": 69.5}
        }
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
    parser.add_argument("--models", nargs="+", choices=[*MODELS])
    return parser


def get_comet_api(api_key=None, **kwargs):
    api_key = api_key or environ.get("COMET_REST_API_KEY")
    api = comet_ml.papi.API(rest_api_key=api_key)
    cached_endpoints = tuple(kwargs.get("cache", ["experiments", "projects"]))
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
            m: [
                get_metric_series(e, "eval_{}".format(m.lower()))
                for e in experiments
                if e.name == name
            ]
            for m in metrics
        }

    return grouped_metrics, metrics


def save_plot(plot, model, plot_metric, **kwargs):
    save_format = kwargs.get("format", "pdf")
    fname = kwargs.get(
        "fname",
        "{}-{}".format(plot_metric, datetime.now().strftime("%d%m%Y_%H%M%S")),
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


def comet_to_df(workspace, models=None, metrics=None, **kwargs):
    cached_data_path = path.join(
        path.dirname(__file__), "temp", "comet_df.pkl"
    )
    use_cached = kwargs.get("use_cached", True)
    if use_cached and path.exists(cached_data_path):
        return unpickle_file(cached_data_path)
    api = get_comet_api(**kwargs)
    workspace = workspace or "reproduction"
    metrics = metrics or ["Macro-F1", "Micro-F1"]
    datasets = kwargs.get("datasets", ["restaurants", "dong", "laptops"])
    embeddings = kwargs.get("embeddings", ["t200", "t100", "cc42", "cc840"])
    other_params = [
        "Train Embeddings",
        "N Hops",
        "Learning Rate",
        "Batch Size",
    ]
    if models is not None and isinstance(models, str):
        models = [models]
    cols = (
        ["Workspace", "Experiment", "Model", "Dataset", "Embedding"]
        + metrics
        + ["Reported {}".format(m) for m in metrics]
        + ["OOV Threshold", "OOV Buckets", "OOV Fn"]
        + other_params
        + ["Vocab Coverage"]
    )
    data = []
    for prj in api.get_projects(workspace):
        if models is not None and prj not in models and not use_cached:
            continue
        grouped_metrics, _ = get_grouped_metric_series(prj, metrics=metrics)
        for exp_name, metric_series in grouped_metrics.items():
            exp = api.get_experiment(workspace, prj, exp_name)
            exp_name_clean = exp.name  # pylint: disable=no-member
            model = exp.project_name  # pylint: disable=no-member
            exp_name_clean = exp_name_clean.replace(model, "")
            dataset = [ds for ds in datasets if ds in exp_name_clean][0]
            exp_name_clean = exp_name_clean.replace(dataset, "")
            embedding = [em for em in embeddings if em in exp_name_clean][0]
            exp_name_clean = exp_name_clean.replace(embedding, "")
            exp_name_clean = exp_name_clean.replace("-", " ").strip()
            params_summary = exp.get_parameters_summary()
            oov_details = sum(
                [
                    [
                        p["valueCurrent"]
                        for p in params_summary
                        if re.match(
                            r"train_(AUTOPARAM: )?Oov {}".format(deet),
                            p["name"],
                        )
                    ]
                    for deet in ["Train", "Buckets", "Fn"]
                ],
                [],
            )
            other_param_values = [
                [
                    p["valueCurrent"]
                    for p in params_summary
                    if re.match(
                        r"train_(AUTOPARAM: )?{}".format(other_param),
                        p["name"],
                    )
                ]
                for other_param in other_params
            ]
            other_param_values = sum(
                [v if len(v) > 0 else [None] for v in other_param_values], []
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

            for run in zip(*(metric_series[m] for m in metrics)):
                data += [
                    [
                        workspace,
                        exp_name_clean,
                        MODELS.get(model, model),
                        dataset.capitalize(),
                        EMBEDDINGS.get(embedding, embedding),
                    ]
                    + [
                        round(max(run_metrics.values()) * 100, 2)
                        for run_metrics in run
                    ]
                    + [
                        REPORTED.get(model, {}).get(dataset, {}).get(m)
                        for m in metrics
                    ]
                    + oov_details
                    + other_param_values
                    + [vocab_coverage]
                ]
    data_frame = pd.DataFrame(data, columns=cols)
    pickle_file(path=cached_data_path, data=data_frame)
    return data_frame


def draw_boxplot(models, **kwargs):
    df = comet_to_df("reproduction", projects=models)
    plot_metrics = kwargs.get("plot_metrics", ["Macro-F1", "Micro-F1"])
    models_tiled = sum([[model] * len(plot_metrics) for model in models], [])
    for (model, plot_metric) in zip(models_tiled, plot_metrics * len(models)):
        dfm = df[(df["Model"] == MODELS.get(model))]
        dss = dfm["Dataset"].unique().tolist()
        num_plots = len(dss)

        handles, labels, colors = [], [], []
        fig, axes = plt.subplots(1, num_plots, sharey=False, figsize=(12, 7))
        fig.suptitle(
            "{}\n({})".format(model.upper(), plot_metric.replace("f1", "F1")),
            fontsize=16,
        )
        fig.subplots_adjust(top=0.85, wspace=(0.2 + (0.025 * num_plots)))
        for i, dataset_name in enumerate(dss):
            try:
                this_ax = axes[i % num_plots]
            except TypeError:
                this_ax = axes
            this_ax.set_title(dataset_name.capitalize())
            this_ax.grid(True, linestyle="dotted", which="both")
            dfmds = dfm[dfm["Dataset"] == dataset_name]
            bp = sns.boxplot(
                y=plot_metric,
                x="Experiment",
                data=dfmds,
                palette="muted",
                hue="Embedding",
                width=0.4,
                ax=this_ax,
                linewidth=1.5,
            )

            this_ax.set_xlabel("")
            this_ax.set_ylabel("")

            original_reported = (
                REPORTED[model].get(dataset_name.lower(), {}).get(plot_metric)
            )
            if original_reported is not None:
                opp_ax = this_ax.twinx()
                opp_ax.tick_params(direction="in", left=True, length=0)
                opp_ax.set_yticks([original_reported])
                opp_ax.set_yticklabels(
                    [original_reported], color="red", va="center"
                )
                opp_ax.get_shared_y_axes().join(this_ax, opp_ax)
                this_ax.axhline(
                    y=original_reported,
                    color="red",
                    linestyle="--",
                    label="Original",
                )
                opp_ax.axhline(
                    y=original_reported,
                    color="red",
                    linestyle="--",
                    label="Original",
                )

            for handle, label in zip(*bp.get_legend_handles_labels()):
                if label not in labels:
                    labels += [label]
                    handles += [handle]
                    if hasattr(handle, "get_facecolor"):
                        colors += [handle.get_facecolor()]

            bp.get_legend().remove()

            cell_text = [
                sum(
                    list(
                        [
                            [],
                            [
                                "{:.2f}%".format(
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
                                "{:.2f}%".format(
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
                [
                    [c[:-1] + tuple([0.5]) for c in colors[: len(cell_group)]]
                    if len(cell_group) > 0
                    else ["none"]
                    for cell_group in cell_row
                ]
                for cell_row in cell_text
            ]
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
                bbox=(0, -0.27, 1, 0.2),
                **({"rowLabels": ["Mean", "Max", "N="]} if i == 0 else {})
            )
            table.set_fontsize(12)

            for (y_pos, x_pos), cell in table.get_celld().items():
                if cell.get_text().get_text() == "":
                    cell.visible_edges = ""
                if x_pos == -1 or y_pos == 2:
                    cell.set_text_props(fontstyle="italic")
                    # pylint: disable=protected-access
                    cell._loc = "right" if x_pos == -1 else cell._loc

            this_ax.set(ylabel=("{}(%)".format(plot_metric) if i == 0 else ""))

        plt.legend(
            handles,
            labels,
            loc="center right",
            title="Legend",
            ncol=1,
            bbox_to_anchor=(1 + (0.44 * len(dss)), 0.5),
        )
        if kwargs.get("fname", False):
            save_plot(plt, model, plot_metric, **kwargs)


def main():
    parser = argument_parser()
    args = parser.parse_args()
    draw_boxplot(args.models)


if __name__ == "__main__":
    main()
