import sys
import pprint
import json
import pickle
from datetime import datetime, timedelta
from os import listdir, system, makedirs, environ
from os.path import isfile, join, dirname
from tempfile import mkdtemp
from shutil import rmtree
from io import BytesIO
from termcolor import colored
from PIL import Image
import docker


def color(key):
    return {
        "r": "red",
        "g": "green",
        "y": "yellow",
        "b": "blue",
        "m": "magenta",
        "c": "cyan",
        "w": "white",
    }.get(key, "grey")


def cprnt(*args, **kwargs):
    output = ""
    indent = " " * int(environ.get("timeit_indent", 0))
    for arg in args:
        kwargs.update({"row": arg})
    for (color_key, string) in kwargs.items():
        if not isinstance(string, str):
            string = pprint.pformat(string)
        col = "".join(filter(str.isalpha, color_key))
        index = col.find("o")
        if index != -1:
            txt, _, frgnd = col.partition("o")
            output += colored(string, color(txt), "on_" + color(frgnd)) + " "
        else:
            output += colored(string, color(col)) + " "
    print(indent + output)


def platform():
    return {
        "linux1": "Linux",
        "linux2": "Linux",
        "darwin": "MacOS",
        "win32": "Windows",
    }.get(sys.platform, sys.platform)


def start_tensorboard(model_dir, port=6006, debug=False, debug_port=6064):
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
    with open(path, "rb") as f:
        return pickle.load(f)


def pickle_file(path, data):
    makedirs(dirname(path), exist_ok=True)
    with open(path, "wb") as pkl_file:
        return pickle.dump(data, pkl_file, pickle.HIGHEST_PROTOCOL)


def dump_json(path, data):
    with open(path, "w+") as json_file:
        json.dump(data, json_file, indent=4)


def search_dir(path, query=None, first=False, kind=None):
    if "file" in kind:
        results = [f for f in listdir(path) if isfile(join(path, f))]
    elif "folder" in kind:
        results = [f for f in listdir(path) if not isfile(join(path, f))]
    else:
        results = [f for f in listdir(path)]
    if query:
        results = [join(path, f) for f in results if query in f]
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


def comet_pretty_log(comet, data_dict, prefix=None, hparams=False):
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
            comet.log_parameter(key, value)
        else:
            comet.log_other(key, value)
