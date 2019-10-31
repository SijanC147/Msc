import sys
import json
import pickle
import math
import csv
import re
import pprint
import subprocess
from math import floor
from zipfile import ZipFile, ZIP_DEFLATED
from datetime import datetime, timedelta
from os import listdir, system, makedirs, environ
from os.path import (
    isfile,
    join,
    dirname,
    basename,
    normpath,
    exists,
    relpath,
    splitext,
    sep,
)
from tempfile import mkdtemp
from shutil import rmtree, copytree, copy as _copy, ignore_patterns
from io import BytesIO
from PIL import Image
import numpy as np
from termcolor import colored
from tensorflow.python.client.timeline import Timeline  # pylint: disable=E0611
from tensorflow.python_io import TFRecordWriter  # noqa
from tensorflow import logging as tf_log
import docker
from tsaplay.constants import NP_RANDOM_SEED, TF_RECORD_SHARDS, PAD_TOKEN
from tsaplay.utils.data import accumulate_dicts


def cprnt(*args, **kwargs):
    colors = {
        "r": "red",
        "g": "green",
        "y": "yellow",
        "b": "blue",
        "m": "magenta",
        "c": "cyan",
        "w": "white",
    }
    output = ""
    grey_output = ""
    for arg in args:
        kwargs.update({"w": arg})
    for (color_key, string) in kwargs.items():
        if color_key in ["end", "tf"]:
            continue
        color_key = {
            "success": "wog",
            "info": "c",
            "train": "b",
            "eval": "r",
            "warn": "wor",
        }.get(color_key.lower(), color_key)
        if not isinstance(string, str):
            string = pprint.pformat(string)
        grey_output += string + " "
        col = "".join(filter(str.isalpha, color_key))
        index = col.find("o")
        if index != -1:
            txt, _, frgnd = col.partition("o")
            output += (
                colored(
                    string,
                    colors.get(txt, "grey"),
                    "on_" + colors.get(frgnd, "grey"),
                )
                + " "
            )
        else:
            output += colored(string, colors.get(col, "grey")) + " "
    if environ.get("TSA_COLORED") == "OFF":
        print(grey_output, end=kwargs.get("end", "\n"))
    else:
        print(output, end=kwargs.get("end", "\n"))
    if environ.get("CLOUD_ML_JOB_ID") is not None:
        tf_level = kwargs.get("tf")
        if tf_level is not None:
            if tf_level in ["DEBUG", "INFO", "WARN", "ERROR", "FATAL"]:
                tf_log.log(
                    getattr(tf_log, kwargs["tf"]),
                    grey_output + kwargs.get("end", "\n"),
                )
            return
        tf_log.TaskLevelStatusMessage(grey_output + kwargs.get("end", "\n"))
        return


def resolve_frequency_steps(freq, epochs=None, epoch_steps=None, default=None):
    if freq is None:
        return epoch_steps if epochs is not None else default
    if epochs is None:
        try:
            return int(freq)
        except ValueError:
            cprnt(
                WARN="Invalid step frequency {}, using default {}".format(
                    freq, default
                )
            )
            return default
    return max(
        1,
        floor(
            epoch_steps
            * ((1 / int(freq[1:])) if str(freq).startswith("/") else int(freq))
        ),
    )


def extract_config_subset(config_objs, keywords):
    config = {}
    if isinstance(keywords, str):
        keywords = [keywords]
    for config_obj in config_objs:
        for keyword in keywords:
            config.update(
                {
                    k.replace(f"{keyword}_", ""): v
                    for k, v in config_obj.items()
                    if k.startswith(f"{keyword}_")
                }
            )
    return config


def platform():
    return {
        "linux1": "Linux",
        "linux2": "Linux",
        "darwin": "MacOS",
        "win32": "Windows",
    }.get(sys.platform, sys.platform)


def start_tensorboard(
    model_dir, port=6006, debug=False, debug_port=6064, sub=None
):
    logdir_str = "--logdir {0} --port {1}".format(model_dir, port)
    debug_str = "--debugger_port {0}".format(debug_port) if debug else ""
    start_tb_cmd = "tensorboard {0} {1}".format(logdir_str, debug_str)

    tb_site = "http://localhost:{0}".format(port)
    open_site_cmd = {
        "Linux": "xdg-open {0}".format(tb_site),
        "Windows": "cmd /c start {0}".format(tb_site),
        "MacOS": "open {0}".format(tb_site),
    }.get(platform())

    if open_site_cmd is not None:
        system(open_site_cmd)

    if sub is not None:
        subprocess.Popen(start_tb_cmd, shell=True)
    else:
        system(start_tb_cmd)


def restart_tf_serve_container():
    logs = None
    client = docker.from_env()
    for container in client.containers.list():
        if container.name == "tsaplay":
            start = datetime.utcnow() - timedelta(seconds=30)
            container.restart()
            logs = container.logs(since=start, stdout=True, stderr=True)
            logs = str(logs, "utf-8")
    return logs


def unpickle_file(path):
    with open(path, "rb") as pkl_file:
        return pickle.load(pkl_file)


def pickle_file(path, data):
    if dirname(path) != "":
        makedirs(dirname(path), exist_ok=True)
    with open(path, "wb") as pkl_file:
        return pickle.dump(data, pkl_file, pickle.HIGHEST_PROTOCOL)


def dump_json(path, data):
    with open(path, "w+") as json_file:
        json.dump(data, json_file, indent=4)


def load_json(path):
    if not exists(path):
        return None
    with open(path, "r") as json_file:
        return json.load(json_file)


def export_run_metadata(run_metadata, path):
    file_name = datetime.now().strftime("%Y%m%d-%H%M%S") + ".json"
    time_line = Timeline(run_metadata.step_stats)  # pylint: disable=E1101
    ctf = time_line.generate_chrome_trace_format()
    write_zippped_file(path=join(path, file_name), data=ctf)


def write_csv(path, data, header=None):
    with open(path, "w", encoding="utf-8") as csvfile:
        if not isinstance(data, dict):
            data = [header] + data if header else data
            writer = csv.writer(csvfile)
            writer.writerows(data)
        else:
            header = header or [*data]
            writer = csv.DictWriter(csvfile, fieldnames=header)
            writer.writeheader()
            for values in zip(*data.values()):
                writer.writerow(
                    {key: value for key, value in zip(header, values)}
                )


def read_csv(path, _format=None):
    if not exists(path):
        return None
    with open(path, "r", encoding="utf-8") as csvfile:
        if isinstance(_format, dict):
            reader = csv.DictReader(csvfile)
            dicts = tuple(
                {k: [v] for k, v in dict(row).items()} for row in reader
            )
            return accumulate_dicts(*dicts)
        reader = csv.reader(csvfile)
        return [row for row in reader]


def write_zippped_file(path, data):
    file_path, file_extension = splitext(path)
    file_name = basename(normpath(path))
    if file_extension != ".zip":
        path = file_path + ".zip"
    with ZipFile(path, "w", ZIP_DEFLATED) as zip_file:
        zip_file.writestr(file_name, data=data)


def search_dir(path, query=None, first=False, kind=None, ci=False):
    kind = kind or []
    if "file" in kind:
        results = [f for f in listdir(path) if isfile(join(path, f))]
    elif "folder" in kind:
        results = [f for f in listdir(path) if not isfile(join(path, f))]
    else:
        results = [f for f in listdir(path)]
    if query:
        results = [
            join(path, f)
            for f in results
            if query in f or (ci and query.lower() in f.lower())
        ]
    return results[0] if first else results


def temp_pngs(images, names):
    temp_dir = mkdtemp()
    try:
        i = 0
        while i < len(names):
            path = join(temp_dir, names[i] + ".png")
            images[i].save(path, optimize=True)
            i += 1
            yield path
    finally:
        rmtree(temp_dir)


def get_image_from_plt(plt):
    with BytesIO() as output:
        try:
            plt.savefig(output, format="png", bbox_inches="tight")
        except ValueError:
            plt.savefig(output, format="png")
        plt.close()
        image_bytes = output.getvalue()

    return Image.open(BytesIO(image_bytes))


def copy(
    src_path, dst_path, rel=None, force=True, ignore=None, file_tree=False
):
    if not isfile(src_path):
        rel_folder = (
            relpath(src_path, rel) if rel else basename(normpath(src_path))
        )
        # dst_path = relpath(dst_path, rel) if rel else dst_path
        dst_path = join(dst_path, rel_folder)
        if exists(dst_path) and not force:
            cprnt(
                warn="COPY: {0} exists, force flag is off so leaving as is.".format(
                    dst_path
                )
            )
            return dst_path
        else:
            rmtree(dst_path, ignore_errors=True)
        ignore_arg = ignore_patterns(ignore) if ignore else None
        copytree(src_path, dst_path, ignore=ignore_arg)
        return dst_path
    if (basename(src_path) != src_path) and file_tree:
        for subdir in src_path.split(sep)[:-1]:
            dst_path = join(dst_path, subdir)
            makedirs(dst_path, exist_ok=force)
    _copy(src_path, dst_path)
    return dst_path


def clean_dirs(*paths):
    for path in paths:
        rmtree(path, ignore_errors=True)
        makedirs(path)


def list_folders(path, kind=None):
    return [
        basename(normpath(folder))
        for folder in search_dir(path, kind=(kind or "folders"))
    ]


def write_tfrecords(path, tf_examples, num_shards=TF_RECORD_SHARDS):
    if NP_RANDOM_SEED is not None:
        np.random.seed(NP_RANDOM_SEED)
    np.random.shuffle(tf_examples)
    tf_examples = [example.SerializeToString() for example in tf_examples]
    num_per_shard = int(math.ceil(len(tf_examples) / float(num_shards)))
    total_shards = int(math.ceil(len(tf_examples) / float(num_per_shard)))
    makedirs(path, exist_ok=True)
    for shard_no in range(total_shards):
        start = shard_no * num_per_shard
        end = min((shard_no + 1) * num_per_shard, len(tf_examples))
        file_name = "{0}_of_{1}.tfrecord".format(shard_no + 1, total_shards)
        file_path = join(path, file_name)
        with TFRecordWriter(file_path) as tf_writer:
            for serialized_example in tf_examples[start:end]:
                tf_writer.write(serialized_example)
        if end == len(tf_examples):
            break


def write_vocab_file(path, vocab, indices=None):
    if indices:
        if PAD_TOKEN not in vocab:
            vocab = [PAD_TOKEN] + vocab
            indices = [0] + indices
        vocab_info = zip(vocab, indices)
        with open(path, "w", encoding="utf-8") as vocab_file:
            for (word, index) in vocab_info:
                vocab_file.write("{0}\t{1}\n".format(word, index))
    else:
        with open(path, "w", encoding="utf-8") as vocab_file:
            for word in vocab:
                vocab_file.write("{0}\n".format(word))


def read_vocab_file(path):
    with open(path, "r", encoding="utf-8") as vocab_file:
        return [word.strip() for word in vocab_file]


def args_to_dict(args):
    args_flat = []
    if not args:
        return {}
    for arg in args:
        if isinstance(arg, str):
            args_flat.append(arg)
        else:
            for sub_arg in arg:
                args_flat.append(sub_arg)
    args_dict = args_flat or {}
    if args_dict:
        args_dict = [arg.split("=") for arg in args_dict]
        args_dict = {
            arg[0]: (
                int(arg[1])
                if arg[1].isdigit()
                else float(arg[1])
                if arg[1].replace(".", "", 1).isdigit()
                else True
                if arg[1].lower() == "true"
                else False
                if arg[1].lower() == "false"
                else arg[1]
            )
            for arg in args_dict
        }
    return args_dict


def arg_with_list(arg):
    pat = re.compile(r"([^\[]*)[\[]?([^\]]*)[\]]?")
    arg = pat.match(arg).groups()
    primary_arg = arg[0]
    additional_info = None
    if arg[1]:
        additional_info = arg[1].split(",")
    return primary_arg, additional_info


def datasets_cli_arg(arg):
    pat = re.compile(r"([^\[]*)[\[]?([^\]]*)[\]]?")
    arg = pat.match(arg).groups()
    primary_arg = arg[0]
    additional_info = None
    if arg[1]:
        additional_info = [
            list(map(float, value.split("/"))) for value in arg[1].split(",")
        ]
        additional_info = (
            {key: val for key, val in zip(["train", "test"], additional_info)}
            if len(additional_info) == 2
            else {"train": additional_info[0], "test": additional_info[0]}
        )
    return primary_arg, additional_info
