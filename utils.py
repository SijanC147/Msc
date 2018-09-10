import spacy
import nltk
import random
import math
import json
import pickle
import numpy as np
import tensorflow as tf
from csv import DictReader, DictWriter
from os import listdir, system, makedirs
from os.path import isfile, join
from functools import wraps
from spacy.attrs import ORTH  # pylint: disable=E0611


def tokenize_phrase(phrase, backend="spacy"):
    if backend == "nltk":
        return nltk.word_tokenize(phrase)
    elif backend == "spacy":
        tokens_list = []
        nlp = spacy.load("en")
        tokens = nlp(str(phrase))
        tokens_list = list(filter(token_filter, tokens))
        return [token.text for token in tokens_list]
    elif backend == "vanilla":
        return phrase.split()


def token_filter(token):
    if token.like_url:
        return False
    if token.like_email:
        return False
    if token.text in ["\uFE0F"]:
        return False
    return True


def re_dist(features, labels, distribution):
    if type(distribution) == list:
        target_dists = distribution
    elif type(distribution) == dict:
        target_dists = [
            distribution["positive"],
            distribution["neutral"],
            distribution["negative"],
        ]

    counts = [
        len([l for l in labels if l == 1]),
        len([l for l in labels if l == 0]),
        len([l for l in labels if l == -1]),
    ]
    target_counts = [0, 0, 0]

    smallest_count_indices = [
        i for i, x in enumerate(counts) if x == min(counts)
    ]
    if len(smallest_count_indices) != 1:
        smallest_count_index = target_dists.index(
            max([target_dists[i] for i in smallest_count_indices])
        )
    else:
        smallest_count_index = smallest_count_indices[0]

    target_counts[smallest_count_index] = counts[smallest_count_index]
    new_total = math.floor(
        counts[smallest_count_index] / target_dists[smallest_count_index]
    )
    counts[smallest_count_index] = float("inf")
    target_dists[smallest_count_index] = float("inf")

    smallest_count_indices = [
        i for i, x in enumerate(counts) if x == min(counts)
    ]
    if len(smallest_count_indices) != 1:
        smallest_count_index = target_dists.index(
            max([target_dists[i] for i in smallest_count_indices])
        )
    else:
        smallest_count_index = smallest_count_indices[0]
    smallest_count_index = counts.index(min(counts))
    target_count = int(new_total * target_dists[smallest_count_index])
    if target_count > counts[smallest_count_index]:
        old_total = new_total
        new_total = math.floor(
            counts[smallest_count_index] / target_dists[smallest_count_index]
        )
        target_counts = [
            math.floor(t * (new_total / old_total)) for t in target_counts
        ]
        target_count = counts[smallest_count_index]
    target_counts[smallest_count_index] = target_count
    counts[smallest_count_index] = float("inf")
    target_dists[smallest_count_index] = float("inf")

    if new_total - sum(target_counts) > min(counts):
        old_total = new_total
        new_total = math.floor(min(counts) / min(target_dists))
        target_counts = [
            math.floor(t * (new_total / old_total)) for t in target_counts
        ]
    target_counts[target_counts.index(0)] = new_total - sum(target_counts)

    postive_sample_indices = [i for i, x in enumerate(labels) if x == 1]
    neutral_sample_indices = [i for i, x in enumerate(labels) if x == 0]
    negative_sample_indices = [i for i, x in enumerate(labels) if x == -1]
    new_features = {
        "sentence": [""] * new_total,
        "sentence_length": [0] * new_total,
        "target": [""] * new_total,
        "mappings": {
            "left": [[]] * new_total,
            "target": [[]] * new_total,
            "right": [[]] * new_total,
        },
    }
    new_labels = [None] * new_total

    for _ in range(target_counts[0]):
        random_index = random.choice(postive_sample_indices)
        random_position = random.randrange(0, new_total)
        while new_labels[random_position] is not None:
            random_position = random.randrange(0, new_total)
        new_features["sentence"][random_position] = features["sentence"][
            random_index
        ]
        new_features["sentence_length"][random_position] = features[
            "sentence_length"
        ][random_index]
        new_features["target"][random_position] = features["target"][
            random_index
        ]
        new_features["mappings"]["left"][random_position] = features[
            "mappings"
        ]["left"][random_index]
        new_features["mappings"]["target"][random_position] = features[
            "mappings"
        ]["target"][random_index]
        new_features["mappings"]["right"][random_position] = features[
            "mappings"
        ]["right"][random_index]
        new_labels[random_position] = labels[random_index]
        postive_sample_indices.remove(random_index)
    for _ in range(target_counts[1]):
        random_index = random.choice(neutral_sample_indices)
        random_position = random.randrange(0, new_total)
        while new_labels[random_position] is not None:
            random_position = random.randrange(0, new_total)
        new_features["sentence"][random_position] = features["sentence"][
            random_index
        ]
        new_features["sentence_length"][random_position] = features[
            "sentence_length"
        ][random_index]
        new_features["target"][random_position] = features["target"][
            random_index
        ]
        new_features["mappings"]["left"][random_position] = features[
            "mappings"
        ]["left"][random_index]
        new_features["mappings"]["target"][random_position] = features[
            "mappings"
        ]["target"][random_index]
        new_features["mappings"]["right"][random_position] = features[
            "mappings"
        ]["right"][random_index]
        new_labels[random_position] = labels[random_index]
        neutral_sample_indices.remove(random_index)
    for _ in range(target_counts[2]):
        random_index = random.choice(negative_sample_indices)
        random_position = random.randrange(0, new_total)
        while new_labels[random_position] is not None:
            random_position = random.randrange(0, new_total)
        new_features["sentence"][random_position] = features["sentence"][
            random_index
        ]
        new_features["sentence_length"][random_position] = features[
            "sentence_length"
        ][random_index]
        new_features["target"][random_position] = features["target"][
            random_index
        ]
        new_features["mappings"]["left"][random_position] = features[
            "mappings"
        ]["left"][random_index]
        new_features["mappings"]["target"][random_position] = features[
            "mappings"
        ]["target"][random_index]
        new_features["mappings"]["right"][random_position] = features[
            "mappings"
        ]["right"][random_index]
        new_labels[random_position] = labels[random_index]
        negative_sample_indices.remove(random_index)

    return new_features, new_labels


def inspect_dist(features, labels):
    positive = [label for label in labels if label == 1]
    neutral = [label for label in labels if label == 0]
    negative = [label for label in labels if label == -1]
    return {
        "num_samples": len(labels),
        "positive": {
            "count": len(positive),
            "percent": round((len(positive) / len(labels)) * 100, 2),
        },
        "neutral": {
            "count": len(neutral),
            "percent": round((len(neutral) / len(labels)) * 100, 2),
        },
        "negative": {
            "count": len(negative),
            "percent": round((len(negative) / len(labels)) * 100, 2),
        },
        "mean_sen_length": round(np.mean(features["sentence_length"]), 2),
    }


def start_tensorboard(model_dir, debug=False):
    logdir = model_dir
    debug_str = "--debugger_port 6064" if debug else ""
    system("open http://localhost:6006")
    system("tensorboard --logdir {dir} {debug}".format(logdir, debug_str))


def default_oov(dim_size):
    return np.random.uniform(low=-0.03, high=0.03, size=dim_size)


def write_stats_to_disk(job, stats, path):
    target_dir = join(path, job)
    makedirs(target_dir, exist_ok=True)
    with open(join(target_dir, "dataset.json"), "w") as file:
        file.write(json.dumps(stats["dataset"]))
    with open(join(target_dir, "job.json"), "w") as file:
        file.write(
            json.dumps(
                {"duration": stats["duration"], "steps": stats["steps"]}
            )
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


def search_dir(dir, query, first=False, files_only=False):
    if files_only:
        results = [
            f for f in listdir(dir) if isfile(join(dir, f)) and query in f
        ]
    else:
        results = [f for f in listdir(dir) if query in f]
    return results[0] if first else results


def corpus_from_docs(docs):
    corpus = {}

    nlp = spacy.load("en")
    tokens = nlp(" ".join(map(lambda document: document.strip(), docs)))
    tokens = list(filter(token_filter, tokens))
    doc = nlp(" ".join(map(lambda token: token.text, tokens)))
    counts = doc.count_by(ORTH)
    words = counts.items()
    for word_id, cnt in sorted(words, reverse=True, key=lambda item: item[1]):
        corpus[nlp.vocab.strings[word_id]] = cnt

    return corpus


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
            f.write(word + "\n")


def get_sentence_contexts(sentence, target, offset=None):
    if offset is None:
        left, _, right = sentence.partition(target)
    else:
        left = sentence[:offset]
        start = offset + len(target)
        right = sentence[start:]
    return left.strip(), right.strip()


def unpickle_file(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def pickle_file(path, data):
    with open(path, "wb") as f:
        return pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
