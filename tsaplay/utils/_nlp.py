import spacy
import numpy as np
from random import choice, randrange
from math import floor
from spacy.attrs import ORTH  # pylint: disable=E0611


def tokenize_phrase(phrase):
    tokens_list = []
    nlp = spacy.load("en", disable=["parser", "ner"])
    tokens = nlp(str(phrase))
    tokens_list = list(filter(token_filter, tokens))
    return [token.text for token in tokens_list]


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
        len([l for l in labels if l == -1]),
        len([l for l in labels if l == 0]),
        len([l for l in labels if l == 1]),
    ]
    target_counts = [0, 0, 0]

    try:
        total_index = target_dists.index(1)
        target_counts[total_index] = counts[total_index]
        new_total = counts[total_index]
    except ValueError:
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
        new_total = floor(
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
            new_total = floor(
                counts[smallest_count_index]
                / target_dists[smallest_count_index]
            )
            target_counts = [
                floor(t * (new_total / old_total)) for t in target_counts
            ]
            target_count = counts[smallest_count_index]
        target_counts[smallest_count_index] = target_count
        counts[smallest_count_index] = float("inf")
        target_dists[smallest_count_index] = float("inf")

        if new_total - sum(target_counts) > min(counts):
            old_total = new_total
            new_total = floor(min(counts) / min(target_dists))
            target_counts = [
                floor(t * (new_total / old_total)) for t in target_counts
            ]
        target_counts[target_counts.index(0)] = new_total - sum(target_counts)
    finally:
        negative_sample_indices = [i for i, x in enumerate(labels) if x == -1]
        neutral_sample_indices = [i for i, x in enumerate(labels) if x == 0]
        positive_sample_indices = [i for i, x in enumerate(labels) if x == 1]
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
            random_index = choice(negative_sample_indices)
            random_position = randrange(0, new_total)
            while new_labels[random_position] is not None:
                random_position = randrange(0, new_total)
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
        for _ in range(target_counts[1]):
            random_index = choice(neutral_sample_indices)
            random_position = randrange(0, new_total)
            while new_labels[random_position] is not None:
                random_position = randrange(0, new_total)
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
            random_index = choice(positive_sample_indices)
            random_position = randrange(0, new_total)
            while new_labels[random_position] is not None:
                random_position = randrange(0, new_total)
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
            positive_sample_indices.remove(random_index)

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


def default_oov(dim_size):
    return np.random.uniform(low=-0.03, high=0.03, size=dim_size)


def corpus_from_docs(docs):
    corpus = {}

    nlp = spacy.load("en", disable=["parser", "ner"])
    docs_joined = " ".join(map(lambda document: document.strip(), docs))
    if len(docs_joined) > 1000000:
        nlp.max_length = len(docs_joined) + 1
    tokens = nlp(docs_joined)
    tokens = list(filter(token_filter, tokens))
    doc = nlp(" ".join(map(lambda token: token.text, tokens)))
    counts = doc.count_by(ORTH)
    words = counts.items()
    for word_id, cnt in sorted(words, reverse=True, key=lambda item: item[1]):
        corpus[nlp.vocab.strings[word_id]] = cnt

    return corpus


def get_sentence_contexts(sentence, target, offset=None):
    if offset is None:
        left, _, right = sentence.partition(target)
    else:
        left = sentence[:offset]
        start = offset + len(target)
        right = sentence[start:]
    return left.strip(), right.strip()