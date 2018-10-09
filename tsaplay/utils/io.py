import sys
import docker
import pprint
from inspect import getsource, getfile
from termcolor import colored
from datetime import datetime, timedelta
from pickle import load, dump, HIGHEST_PROTOCOL
from os.path import isfile, join, dirname, exists
from json import dumps
from os import listdir, system, makedirs
from tempfile import mkdtemp
from shutil import rmtree
from io import BytesIO
from PIL import Image
from csv import DictReader, DictWriter
from contextlib import contextmanager


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
    for arg in args:
        kwargs.update({"row": arg})
    for (c, string) in kwargs.items():
        if not isinstance(string, str):
            string = pprint.pformat(string)
        col = "".join(filter(str.isalpha, c))
        index = col.find("o")
        if index != -1:
            txt, _, frgnd = col.partition("o")
            output += colored(string, color(txt), "on_" + color(frgnd)) + " "
        else:
            output += colored(string, color(col)) + " "
    print(output)


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
        return load(f)


def pickle_file(path, data):
    makedirs(dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        return dump(data, f, HIGHEST_PROTOCOL)


def corpus_from_csv(path):
    corpus = {}
    with open(path) as csvfile:
        reader = DictReader(csvfile)
        for row in reader:
            corpus[row["word"]] = int(row["count"])
    return corpus


def corpus_to_csv(path, corpus):
    with open(path, "w") as csvfile:
        writer = DictWriter(csvfile, fieldnames=["word", "count"])
        writer.writeheader()
        for word, count in corpus.items():
            row = {"word": word, "count": count}
            writer.writerow(row)


def search_dir(dir, query, first=False, files_only=False):
    if files_only:
        results = [
            join(dir, f)
            for f in listdir(dir)
            if isfile(join(dir, f)) and query in f
        ]
    else:
        results = [join(dir, f) for f in listdir(dir) if query in f]
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
