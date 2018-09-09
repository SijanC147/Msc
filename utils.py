import os
import spacy
import nltk
import random
import math
import numpy as np
import tensorflow as tf


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


def re_distribute(features, labels, distribution):
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


def inspect_distribution(features, labels):
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
    data = {
        "logdir": model_dir,
        "debug": "--debugger_port 6064" if debug else "",
    }
    os.system("open http://localhost:6006")
    os.system("tensorboard --logdir {dir} {debug}".format(data))


def default_oov(dim_size):
    return np.random.uniform(low=-0.03, high=0.03, size=dim_size)
