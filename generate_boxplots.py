# pylint: disable=too-many-locals,too-many-statements,wrong-import-position
import warnings
import argparse
from os import path, makedirs, getcwd, environ
import re
from datetime import datetime
import json
from itertools import groupby
from operator import itemgetter
from collections import OrderedDict
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
        "Moore et al. 2018 (Mean)": {"dong": {"Macro-F1": 60.69}},
        "Moore et al. 2018 (Max)": {"dong": {"Macro-F1": 64.34}},
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
        "Moore et al. 2018 (Mean)": {"dong": {"Macro-F1": 65.63}},
        "Moore et al. 2018 (Max)": {"dong": {"Macro-F1": 67.04}},
    },
    "tclstm": {
        # ! Scores from original paper
        "Tang et al. 2016 (Original)": {
            "dong": {"Micro-F1": 71.5, "Macro-F1": 69.5}
        },
        # ! Scores from Reproduction paper
        "Moore et al. 2018 (Mean)": {"dong": {"Macro-F1": 65.23}},
        "Moore et al. 2018 (Max)": {"dong": {"Macro-F1": 67.66}},
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
            "laptops": {"Micro-F1": 72.1},
            "restaurants": {"Micro-F1": 78.6},
        },
        # # ! Scores from LCR-ROT
        # "Zheng et al. 2018": {
        #     "laptops": {"Micro-F1": 72.1},
        #     "restaurants": {"Micro-F1": 78.6},
        # },
        # ! Scores from https://arxiv.org/abs/2005.06607
        "Navonil et al. 2020": {
            "laptops": {"Macro-F1": 64.86},
            "restaurants": {"Macro-F1": 66.41},
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
    "Dataset": {
        "cmt_regex": r"{}",
        "cmt_key": "datasets",
        "df_valfn": lambda v: str(v).capitalize(),
        "use_xlabel": False,
    },
    "Class Labels": {
        "cmt_regex": r"datasets_{}",
        "cmt_key": "all_class_labels",
    },
    "Train Distribution": {
        "cmt_regex": r"datasets_{}",
        "cmt_key": "train_dist",
        "use_xlabel": False,
        "incl_xaxis_label": False,
    },
    "Train Balanced": {
        "cmt_regex": r"datasets_{}",
        "cmt_key": "train_dist",
        "df_valfn": lambda v: (v in ["34/33/33", "33/34/33", "33/33/34"]),
        "xlabel_fn": lambda v: "B" if v else "N",
    },
    "Test Distribution": {"cmt_regex": r"datasets_{}", "cmt_key": "test_dist"},
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
        "df_valfn": lambda v: {"GradientDescent": "SGD"}
        .get(v.replace("Optimizer", ""), v.replace("Optimizer", ""))
        .capitalize(),
        "xlabel_fn": lambda v: {
            "momentum": "mnt",
            "adagrad": "adg",
            "adam": "adm",
            "sgd": "sgd",
        }.get(v.lower(), v),
    },
    "Kernel Initializer": {"cmt_key": "Initializer"},
    "Bias Initializer": {"cmt_key": "Bias Initializer"},
    "Batch Size": {"cmt_key": "Batch Size"},
    "Epochs Trained": {
        "cmt_key": "curr_epoch",
        "use_xlabel": False,
        "df_valfn": int,
    },
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

# EMBEDDINGS = {
#     "cc42": "GloVe CommonCrawl 42b (300d)",
#     "cc840": "GloVe CommonCrawl 840b (300d)",
#     "t200": "GloVe Twitter (200d)",
#     "t100": "GloVe Twitter (100d)",
# }

EMBEDDINGS = {
    "cc840": "300d 840b CommonCrawl GloVe ",
    "cc42": "300d 42b CommonCrawl GloVe",
    "t200": "200d Twitter GloVe",
    "t100": "100d Twitter GloVe",
}

METRIC_COLS = {
    "rep": {
        # pylint: disable=unnecessary-lambda
        "df_keyfn": lambda m: "Reported {}".format(m),
        "df_valfn": lambda m, x: REPORTED.get(
            {v: k for k, v in MODELS.items()}.get(x.Model), {}
        )
        .get(
            (
                [
                    k
                    for k in [
                        *REPORTED.get(
                            {v: k for k, v in MODELS.items()}.get(x.Model), {}
                        )
                    ]
                    if "original" in k.lower()
                ]
                or ["none"]
            )[0],
            {},
        )
        .get(x.Dataset.lower(), {})
        .get(m),
    },
    "max": {
        # pylint: disable=unnecessary-lambda
        "df_keyfn": lambda m: "Max {}".format(m),
        "df_valfn": lambda m, x: round(max(x[m].values()) * 100, 2),
    },
}

CMT_METRICS_MAPPING = {
    "Macro-F1": {
        "cmt_keyfn": lambda ctxs=["eval"]: [
            "{}_Macro-F1".format(ctx).lower() for ctx in ctxs
        ],
        "df_keyfn": lambda: ["Macro-F1"],
        "incl_cols": ["rep", "max"],
    },
    "Micro-F1": {
        "cmt_keyfn": lambda ctxs=["eval"]: [
            "{}_Micro-F1".format(ctx).lower() for ctx in ctxs
        ],
        "df_keyfn": lambda: ["Micro-F1"],
        "incl_cols": ["rep", "max"],
    },
    "Loss": {
        "cmt_keyfn": lambda ctxs=["train", "eval"]: [
            "{}_Loss".format(ctx).lower() for ctx in ctxs
        ],
        "df_keyfn": lambda ctxs=["train", "eval"]: [
            "{} Loss".format(ctx.capitalize()) for ctx in ctxs
        ],
    },
}


CACHE_DATA_PATH = path.join(path.dirname(__file__), "temp", "comet_dataframes")


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


# DEPRECATED, need to use metrics_for_chart now
# def get_metric_series(experiment, metric_cmt_key):
#     return {
#         v["epoch"]: float(v["metricValue"])
#         for v in experiment.get_metrics()
#         if v["metricName"] == metric_cmt_key
#     }


def get_metric_series(experiment, metric_cmt_key, api):
    series_data_full = api.get_metrics_for_chart(
        experiment_keys=[experiment.id], metrics=[metric_cmt_key]
    )
    metric_series_data = [
        {
            ep: float(val)
            for (ep, val) in zip(metrics["epochs"], metrics["values"])
        }
        for metrics in series_data_full[experiment.id]["metrics"]
        if metrics["metricName"] == metric_cmt_key
    ]
    return metric_series_data[0]


def get_grouped_metric_series(project, metrics, workspace=None, **kwargs):
    api = get_comet_api(**kwargs)
    workspace = workspace or "reproduction-new"
    grouped_metrics = {}
    experiments = api.get_experiments(workspace, project_name=project)
    unique_experiment_names = np.unique(
        [
            e.name
            for e in experiments
            if e.name is not None and not e.name.startswith("#")
        ]
    ).tolist()
    metrics_cmt_keys = sum(
        [CMT_METRICS_MAPPING[m]["cmt_keyfn"]() for m in metrics], []
    )
    for name in unique_experiment_names:
        grouped_metrics[name] = OrderedDict(
            {
                "experiments": [e for e in experiments if e.name == name],
                **{
                    metric_cmt_key: [
                        get_metric_series(e, metric_cmt_key, api)
                        for e in experiments
                        if e.name == name
                    ]
                    for metric_cmt_key in metrics_cmt_keys
                },
            }
        )

    return grouped_metrics, metrics


def save_plot(plot, model, plot_metric, **kwargs):
    save_format = kwargs.get("format", "pdf")
    name = kwargs.get("fname")
    fname = "{}{}-{}".format(
        ((name + "-") if name else ""),
        plot_metric,
        datetime.now().strftime("%d.%m.%Y_%H.%M.%S"),
    )
    target_dir = path.join(
        kwargs.get("path", getcwd()),
        "figures",
        "reproduciton",
        MODELS.get(model, model),
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
        CACHE_DATA_PATH,
        f"{'all' if models is None else '_'.join(sorted(models))}.pkl",
    )
    use_cached = kwargs.get("use_cached", False)
    if use_cached and path.exists(cached_data_path):
        return unpickle_file(cached_data_path)
    api = get_comet_api(**kwargs)
    workspace = workspace or "reproduction"
    metrics = metrics or ["Macro-F1", "Micro-F1", "Loss"]
    metric_series_cols = sum(
        [CMT_METRICS_MAPPING[m]["df_keyfn"]() for m in metrics], []
    )
    cols = (
        ["ID", "Workspace", "Experiment", "Model", "Embedding"]
        + metric_series_cols
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
        exp_groups, _ = get_grouped_metric_series(prj, metrics, workspace)
        for _, exp_group_runs in exp_groups.items():
            # ? (exp, *run_metrics)->(APIExperiment,([{epoch: val}∀{ctx,met}]))
            for (exp, *run_metrics) in zip(*exp_group_runs.values()):
                if kwargs.get("excl_curr", True) and is_currently_running(exp):
                    continue
                model = exp.project_name  # pylint: disable=no-member
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
                            cmt_param_other.get(
                                "cmt_regex", r"train_(AUTOPARAM: )?{}"
                            ).format(cmt_param_other["cmt_key"]),
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

                exp_info = json.loads(
                    exp.get_asset(
                        asset_id=[
                            f["assetId"]
                            for f in exp.get_asset_list()
                            if f["fileName"] == "info.json"
                        ][0],
                        return_type="text",
                    )
                )

                vocab_coverage = exp_info.get("vocab_coverage", {})
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

                datasets_info = exp_info["datasets"]
                for dataset in datasets_info.values():
                    ds_name = dataset["name"]

                embedding_info = exp_info["embedding"]
                embedding_str = {
                    "fasttext-wiki-news-subwords-300": "FastText (300d)",
                    "glove-twitter-25": "GloVe Twitter (25d)",
                    "glove-twitter-50": "GloVe Twitter (50d)",
                    "glove-twitter-100": EMBEDDINGS["t100"],
                    "glove-twitter-200": EMBEDDINGS["t200"],
                    "glove-wiki-gigaword-50": "GloVe Wiki (50d)",
                    "glove-wiki-gigaword-100": "GloVe Wiki (100d)",
                    "glove-wiki-gigaword-200": "GloVe Wiki (200d)",
                    "glove-wiki-gigaword-300": "GloVe Wiki (300d)",
                    "glove-cc42-300": EMBEDDINGS["cc42"],
                    "glove-cc840-300": EMBEDDINGS["cc840"],
                    "word2vec-google-news-300": "Word2Vec Google News (300d)",
                    "word2vec-ruscorpora-300": "Word2Vec Rus Corpora (300d)",
                }.get(embedding_info["name"])

                exp_name_str = exp.name  # pylint: disable=no-member
                exp_name_str = exp_name_str.replace(model, "")
                exp_name_str = exp_name_str.replace(ds_name, "")
                exp_name_str = exp_name_str.replace("balanced", "")
                exp_name_str = exp_name_str.replace(
                    {v: k for k, v in EMBEDDINGS.items()}.get(embedding_str),
                    # {
                    #     "GloVe CommonCrawl 42b (300d)": "cc42",
                    #     "GloVe CommonCrawl 840b (300d)": "cc840",
                    #     "GloVe Twitter (100d)": "t100",
                    #     "GloVe Twitter (200d)": "t200",
                    # }.get(embedding_str),
                    "",
                )
                exp_name_str = exp_name_str.replace("-", " ")
                exp_name_str = exp_name_str.strip()

                data += [
                    [
                        exp.id,
                        workspace,
                        exp_name_str,
                        MODELS.get(model, model),
                        embedding_str,
                    ]
                    + run_metrics
                    + cmt_params_others_values
                    + vocab_data
                ]
    data_frame = pd.DataFrame(data, columns=cols)
    for metric in metrics:
        for incl_col in CMT_METRICS_MAPPING[metric].get("incl_cols", []):
            mcol_map = METRIC_COLS[incl_col]
            for m_col in CMT_METRICS_MAPPING[metric]["df_keyfn"]():
                col_name = mcol_map["df_keyfn"](m_col)
                col_fn = lambda x, i=mcol_map, m=m_col: i["df_valfn"](m, x)
                data_frame[col_name] = data_frame.apply(col_fn, axis=1)
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


def xaxis_order(order_by=None):
    order_by = order_by or {
        "Hidden Units": "desc",
        "Train Balanced": "desc",
        "Optimizer": "asc",
        "Learning Rate": "asc",
    }
    if isinstance(order_by, list):
        order_by = {"by": order_by}
    elif isinstance(order_by, dict):
        order_by = {
            "by": [*order_by],
            "ascending": [v == "asc" for v in order_by.values()],
        }
    return order_by


def filter_df(data, filter_dict):
    single_filters = {
        k: v for k, v in filter_dict.items() if not isinstance(v, list)
    }
    dff = (
        data.loc[
            (data[[*single_filters]] == pd.Series(single_filters)).all(axis=1)
        ]
        if single_filters
        else data
    )
    multi_filters = {
        k: v for k, v in filter_dict.items() if isinstance(v, list)
    }
    for k, vals in multi_filters.items():
        dff = dff[(dff[k]).isin(vals)]
    return dff


def exp_per_dataset(data, ref_width=10):
    expcnt = {
        ds: len(data[(data["Dataset"] == ds)]["Experiment"].unique().tolist())
        for ds in data["Dataset"].unique().tolist()
    }
    width_ratios = [
        round((v / sum(expcnt.values())) * ref_width) for v in expcnt.values()
    ] + [1]
    return expcnt, width_ratios


def cmn_diff_hparams(data, hparams=None, order_by=None):
    hparams = hparams or [*CMT_VALS_MAPPING]
    hparams = [
        hp
        for hp in hparams
        if hp in [*data] and data[hp].unique().tolist()[0] is not None
    ]
    common_hparams = {
        hp: str(data[hp].unique().tolist()[0])
        for hp in hparams
        if len(data[hp].unique().tolist()) == 1
    }
    diff_hparams = [hp for hp in hparams if hp not in [*common_hparams]]
    if order_by:
        diff_hparams = [dhp for dhp in order_by if dhp in diff_hparams] + [
            dhp for dhp in diff_hparams if dhp not in order_by
        ]
    return common_hparams, diff_hparams


def val_or_dict_entry(data, key):
    if isinstance(data, dict) and key in [*data]:
        return data[key]
    return data


def x_template_and_factors(templates, hparams, model):
    template = val_or_dict_entry(templates, model)
    if template:
        factors = re.findall(r"\{([^\}]*)?", template)
    else:
        factors = [
            hp
            for hp in hparams
            if CMT_VALS_MAPPING[hp].get("use_xlabel", True)
        ]
        template = "\n".join(
            [
                CMT_VALS_MAPPING[factor].get("xtick_label", False)
                or "{{{}}}".format(factor)
                for factor in factors
            ]
        )
    return template, factors


def format_xtick_labels(data, labels, template):
    label_attribute = lambda k, v: (
        CMT_VALS_MAPPING.get(k, {}).get("xlabel_fn")(v)
        if CMT_VALS_MAPPING.get(k, {}).get("xlabel_fn") is not None
        else v
        if v is not None
        else "-"
    )
    xticklabels = [
        template.format_map(
            {
                k: label_attribute(k, v)
                for k, v in filter_df(data, {"Experiment": lab})
                .to_dict("records")[0]
                .items()
            }
        )
        for lab in labels
    ]
    return xticklabels


def draw_grid_series(data, model, filters=None, **kwargs):
    metrics = kwargs.get("metrics", ["Train Loss", "Eval Loss", "Macro-F1"])
    bands = kwargs.get("bands", "minmax")
    title = "{} ({})".format(kwargs.get("title", model.upper()), bands)
    data["Experiment"] = data.apply(
        lambda x: " ".join([x["Experiment"], x["Dataset"], x["Embedding"]]),
        axis=1,
    )
    data["Embedding(Short)"] = data.apply(
        lambda x: {v: k for k, v in EMBEDDINGS.items()}.get(x["Embedding"]),
        axis=1,
    )
    data = filter_df(data, {"Model": model, **(filters or {})})
    common, hparams = cmn_diff_hparams(data)
    hparams = ["Embedding(Short)"] + hparams
    exp_groups = data["Experiment"].unique().tolist()
    height = max(3 * len(exp_groups), 8)
    fig, axes = plt.subplots(
        len(exp_groups) + 2, len(metrics) + 1, figsize=(16, height)
    )
    cmn_gs = axes[2, 2].get_gridspec()
    for row in [0, 1]:
        for ax in axes[row, :]:
            ax.remove()
    cmn_ax = fig.add_subplot(cmn_gs[:2, :])
    cmn_ax.axis("off")
    cmn_ax.set_title(title)
    cmn_ax.table(
        cellText=[[k, v] for k, v in common.items()],
        cellLoc="center",
        edges="open",
        bbox=(0, 0, 1, 1),
    )
    for ax_y, group_name in enumerate(exp_groups):
        group_data = data[(data["Experiment"] == group_name)]
        info_ax = axes[ax_y + 2, 0]
        info_ax.axis("off")
        table_data = [["Count", len(group_data)]]
        table_data += [
            [hp, group_data[hp].unique().tolist()[0]]
            for hp in hparams
            if (
                len(group_data[hp].unique().tolist()) == 1
                and group_data[hp].unique().tolist()[0]
            )
        ]
        info_table = info_ax.table(
            cellText=table_data, edges="open", bbox=(0, 0, 1, 1)
        )
        if kwargs.get("info_font", False):
            info_table.auto_set_font_size(False)
            info_table.set_fontsize(kwargs.get("info_font"))
        for ax_x, metric in enumerate(metrics):
            this_ax = axes[ax_y + 2, ax_x + 1]
            if ax_y == 0:
                this_ax.set_title(metric)
            this_ax.grid(True, linestyle="dotted", which="both")
            group_metric = group_data[metric]
            longest_train = max([len(gm) for gm in group_metric])
            epochs = range(longest_train)
            series_data = []
            for i in range(longest_train):
                series_data += [
                    np.array(
                        [
                            list(dict(gm).values())[i]
                            for gm in group_metric
                            if i < len(list(dict(gm).values()))
                        ]
                    )
                ]
            mean_metric = np.array([np.mean(e_d) for e_d in series_data])
            if bands == "stddev":
                band_min = np.array(
                    [
                        m - np.std(e_d)
                        for m, e_d in zip(mean_metric, series_data)
                    ]
                )
                band_max = np.array(
                    [
                        m + np.std(e_d)
                        for m, e_d in zip(mean_metric, series_data)
                    ]
                )
            else:
                band_min = np.array([np.min(e_d) for e_d in series_data])
                band_max = np.array([np.max(e_d) for e_d in series_data])
            sns.lineplot(epochs, mean_metric, ax=this_ax)
            this_ax.fill_between(epochs, band_min, band_max, alpha=0.3)
    if kwargs.get("fname", False) or kwargs.get("format", False):
        save_plot(plt, model, "Series-{}".format(bands), **kwargs)
    else:
        plt.show()


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
    workspace = kwargs.get("workspace", "reproduction-new")
    df = comet_to_df(workspace, models=models, **kwargs)
    draw_boxplot.df = df.copy()
    plot_metrics = kwargs.get("plot_metrics", ["Macro-F1", "Micro-F1"])
    xticklabel_templates = kwargs.get("xticklabel_templates", dict())
    models = models or [*MODELS]
    models_tiled = sum([[model] * len(plot_metrics) for model in models], [])
    box_palette = kwargs.get("box_palette", "muted")
    order_by = xaxis_order(kwargs.get("order_by"))
    for (model, plot_metric) in zip(models_tiled, plot_metrics * len(models)):
        plot_metric_dfcol = "Max {}".format(plot_metric)
        df_filters = {**kwargs.get("filters", {}), "Model": MODELS.get(model)}
        dfm = filter_df(df, df_filters)
        if dfm.empty:
            warnings.warn("EMPTY DATAFRAME: {}({})".format(model, plot_metric))
            continue
        common_hparams, diff_hparams = cmn_diff_hparams(
            dfm, order_by=order_by["by"]
        )
        xticklabel_template, xticklabel_factors = x_template_and_factors(
            xticklabel_templates, diff_hparams, model
        )
        expcnt, width_ratios = exp_per_dataset(dfm, kwargs.get("width", 10))
        num_plots, dss = len(expcnt), [*expcnt]
        handles, labels = [], []
        legends, colors = dict(), dict()
        fig, axes = plt.subplots(
            1,
            num_plots + 1,
            sharey=False,
            figsize=(16, 7),
            gridspec_kw={"width_ratios": width_ratios},
        )
        fig.suptitle(
            "{}\n({})".format(model.upper(), plot_metric.replace("f1", "F1")),
            fontsize=14,
        )
        fig.subplots_adjust(top=0.85, wspace=(0.4 + (0.025 * num_plots)))
        for i, dataset_name in enumerate(dss):
            try:
                this_ax = axes[i % num_plots]
            except TypeError:
                this_ax = axes

            cited_results = group_citations(dataset_name, plot_metric, model)
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

            this_ax.set_title(dataset_name.capitalize().replace("[", "\n["))
            this_ax.grid(True, linestyle="dotted", which="both")
            dfmds = filter_df(dfm, {"Dataset": dataset_name})
            box_order = (
                dfmds.sort_values(**order_by)["Experiment"].unique().tolist()
            )
            bp = sns.boxplot(
                y=plot_metric_dfcol,
                x="Experiment",
                hue="Embedding",
                order=box_order,
                data=dfmds,
                palette=box_palette,
                ax=this_ax,
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
            for handle, label, box_artist in zip(
                *bp.get_legend_handles_labels(), bp.artists
            ):
                if label not in labels:
                    labels += [label]
                    handles += [handle]
                    if hasattr(handle, "get_facecolor"):
                        colors[label] = handle.get_facecolor()
                else:
                    box_artist.set_facecolor(colors[label])
            sp = (
                sns.stripplot(
                    y=plot_metric_dfcol,
                    x="Experiment",
                    hue="Embedding",
                    order=box_order,
                    jitter=True,
                    data=dfmds,
                    ax=this_ax,
                    dodge=True,
                    palette=sns.set_palette(
                        sns.color_palette(
                            [a.get_facecolor() for a in bp.artists]
                        )
                    ),
                    edgecolor="grey",
                )
                if kwargs.get("strip", False)
                else None
            )

            label_xaxis = kwargs.get("label_xaxis") or bool(
                xticklabel_template
            )
            if label_xaxis:
                current_labels = [
                    l.get_text() for l in this_ax.get_xticklabels()
                ]
                new_labels = format_xtick_labels(
                    dfmds, current_labels, xticklabel_template
                )
                this_ax.set_xticklabels(new_labels)
            else:
                this_ax.set_xticks([])

            xaxis_labels_height = -1 * (
                (kwargs.get("xoff", 0.15) * len(xticklabel_factors))
                if label_xaxis
                else 0.25
            )
            this_ax.set_xlabel("")
            this_ax.set_ylabel("")
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
                                        ][plot_metric_dfcol].mean(),
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
                                    ][plot_metric_dfcol].max()
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

            sep_width = kwargs.get("sepw", 0.03)
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
            table.set_fontsize(8)

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
            col_widths = [0.1 * num_plots, 1 - (0.1 * num_plots)]
            hparam_table = last_axis.table(
                cellText=[["Hyper-Parameter Configuration\n", ""]]
                + [[k, v] for k, v in common_hparams.items()],
                colWidths=col_widths,
                cellLoc="left",
                bbox=(-0.2, xaxis_labels_height, 2, 0.3),
                edges="open",
            )
            hparam_table.auto_set_font_size(False)
            hparam_table.set_fontsize(7)
            for (y_pos, x_pos), cell in hparam_table.get_celld().items():
                if x_pos == 0 and y_pos == 0:
                    cell.set_text_props(fontsize="x-small")
                if x_pos == 1:
                    # pylint: disable=protected-access
                    cell._loc = "right"

        if kwargs.get("fname", False) or kwargs.get("format", False):
            save_plot(plt, model, plot_metric, **kwargs)
        else:
            plt.show()


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models", "-m", nargs="+", choices=[*MODELS], default=None
    )
    parser.add_argument("--params", "-p", nargs="*", default=None)
    return parser


def main():
    parser = argument_parser()
    args = parser.parse_args()
    params = args_to_dict(args.params)
    draw_boxplot(models=args.models, **params)


if __name__ == "__main__":
    main()
