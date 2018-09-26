import sys
import docker
import time
from termcolor import colored
from datetime import datetime, timedelta
from pickle import load, dump, HIGHEST_PROTOCOL
from os.path import isfile, join, dirname
from json import dumps
from os import listdir, system, makedirs
from csv import DictReader, DictWriter


def get_platform():
    return {
        "linux1": "Linux",
        "linux2": "Linux",
        "darwin": "MacOS",
        "win32": "Windows",
    }.get(sys.platform, sys.platform)


def gprint(string):
    print(colored(string, "green"))


def start_tensorboard(model_dir, port=6006, debug=False, debug_port=6064):
    logdir_str = "--logdir {0} --port {1}".format(model_dir, port)
    debug_str = "--debugger_port {0}".format(debug_port) if debug else ""
    start_tb_cmd = "tensorboard {0} {1}".format(logdir_str, debug_str)

    tb_site = "http://localhost:{0}".format(port)
    open_site_cmd = {
        "Linux": "xdg-open {0}".format(tb_site),
        "Windows": "cmd /c start {0}".format(tb_site),
        "MacOS": "open {0}".format(tb_site),
    }.get(get_platform())

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


def write_embedding_to_disk(path, emb_dict):
    with open(path, "w+") as f:
        for word in [*emb_dict]:
            if word != "<OOV>" and word != "<PAD>":
                vector = " ".join(emb_dict[word].astype(str))
                f.write("{w} {v}\n".format(w=word, v=vector))


def write_emb_tsv_to_disk(path, emb_dict):
    with open(path, "w+") as f:
        f.write("Words\n")
        for word in [*emb_dict]:
            if word != "<OOV>" and word != "<PAD>":
                f.write(word + "\n")


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


def write_stats_to_disk(job, stats, path):
    target_dir = join(path, job)
    makedirs(target_dir, exist_ok=True)
    with open(join(target_dir, "dataset.json"), "w") as file:
        file.write(dumps(stats["dataset"]))
    with open(join(target_dir, "job.json"), "w") as file:
        file.write(
            dumps({"duration": stats["duration"], "steps": stats["steps"]})
        )
    with open(join(target_dir, "model.md"), "w") as file:
        file.write("## Model Params\n")
        file.write("````Python\n")
        file.write(str(stats["model"]["params"]) + "\n")
        file.write("````\n")
        file.write("## Train Input Fn\n")
        file.write("````Python\n")
        file.write(str(stats["model"]["train_input_fn"]) + "\n")
        file.write("````\n")
        file.write("## Eval Input Fn\n")
        file.write("````Python\n")
        file.write(str(stats["model"]["eval_input_fn"]) + "\n")
        file.write("````\n")
        file.write("## Model Fn\n")
        file.write("````Python\n")
        file.write(str(stats["model"]["model_fn"]) + "\n")
        file.write("````\n")
    with open(join(target_dir, "estimator.md"), "w") as file:
        file.write("## Train Hooks\n")
        file.write("````Python\n")
        file.write(str(stats["estimator"]["train_hooks"]) + "\n")
        file.write("````\n")
        file.write("## Eval Hooks\n")
        file.write("````Python\n")
        file.write(str(stats["estimator"]["eval_hooks"]) + "\n")
        file.write("````\n")
        file.write("## Train Fn\n")
        file.write("````Python\n")
        file.write(str(stats["estimator"]["train_fn"]) + "\n")
        file.write("````\n")
        file.write("## Eval Fn\n")
        file.write("````Python\n")
        file.write(str(stats["estimator"]["eval_fn"]) + "\n")
        file.write("````\n")
        file.write("## Train And Eval Fn\n")
        file.write("````Python\n")
        file.write(str(stats["estimator"]["train_eval_fn"]) + "\n")
        file.write("````\n")
    if len(stats["common"]) > 0:
        with open(join(target_dir, "common.md"), "w") as file:
            file.write("## Model Common Functions\n")
            file.write("````Python\n")
            file.write(str(stats["common"]) + "\n")
            file.write("````\n")

