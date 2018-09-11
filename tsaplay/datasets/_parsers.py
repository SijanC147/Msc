from csv import DictReader
from json import loads
from itertools import chain
from collections import defaultdict
from os.path import join, dirname


def dong_parser(path):
    lines = open(path, "r").readlines()
    sentences = [
        lines[index].strip().replace("$T$", lines[index + 1].strip())
        for index in range(0, len(lines), 3)
    ]
    targets = [lines[index + 1].strip() for index in range(0, len(lines), 3)]
    labels = [
        int(lines[index + 2].strip()) for index in range(0, len(lines), 3)
    ]
    return sentences, targets, labels


def nakov_parser(path):
    sentences = []
    targets = []
    labels = []
    with open(path, "r") as file:
        reader = DictReader(
            file,
            dialect="excel-tab",
            fieldnames=["tweet_id", "target", "sentiment", "sentence"],
        )
        for row in reader:
            sentences.append(row["sentence"])
            targets.append(row["target"])
            labels.append(
                {"2": 1, "1": 1, "0": 0, "-1": -1, "-2": -1}.get(
                    row["sentiment"]
                )
            )
    return sentences, targets, labels


def rosenthal_parser(path):
    sentences = []
    targets = []
    labels = []
    with open(path, "r") as file:
        reader = DictReader(
            file,
            dialect="excel-tab",
            fieldnames=["tweet_id", "target", "sentiment", "sentence"],
        )
        for row in reader:
            if row["sentiment"] in ["positive", "neutral", "negative"]:
                sentences.append(row["sentence"])
                targets.append(row["target"])
                labels.append(
                    {"positive": 1, "neutral": 0, "negative": -1}.get(
                        row["sentiment"]
                    )
                )
    return sentences, targets, labels


def saeidi_parser(path):
    data = loads(open(path, "r").read())
    sentences = [
        j
        for i in [
            [sample["text"]]
            * len(
                [
                    opinion
                    for opinion in sample["opinions"]
                    if opinion["aspect"] == "general"
                ]
            )
            for sample in data
        ]
        for j in i
    ]
    targets = [
        j
        for i in [
            [
                opinion["target_entity"]
                for opinion in sample["opinions"]
                if opinion["aspect"] == "general"
            ]
            for sample in data
        ]
        for j in i
    ]
    labels = [
        j
        for i in [
            [
                {"Positive": 1, "Negative": 0}.get(opinion["sentiment"])
                for opinion in sample["opinions"]
                if opinion["aspect"] == "general"
            ]
            for sample in data
        ]
        for j in i
    ]
    return sentences, targets, labels


def wang_parser(path):
    lines = open(path, "r").readlines()
    tweets = []
    annotations = []
    for line in lines:
        tweets.append(
            loads(
                open(
                    join(
                        dirname(path), "tweets", "5" + line.strip() + ".json"
                    ),
                    "r",
                ).read()
            )
        )
        annotations.append(
            loads(
                open(
                    join(
                        dirname(path),
                        "annotations",
                        "5" + line.strip() + ".json",
                    ),
                    "r",
                ).read()
            )
        )

    tweets_and_annotations = defaultdict(lambda: {})
    for sample in chain(tweets, annotations):
        tweets_and_annotations[sample["tweet_id"]].update(sample)

    data = tweets_and_annotations.values()

    sentences = [
        j
        for i in [
            [sample["content"]]
            * len(
                [
                    entity
                    for entity in sample["entities"]
                    if sample["items"][str(entity["id"])]
                    in ["positive", "neutral", "negative"]
                ]
            )
            for sample in data
        ]
        for j in i
    ]
    targets = [
        j
        for i in [
            [
                entity["entity"]
                for entity in sample["entities"]
                if sample["items"][str(entity["id"])]
                in ["positive", "neutral", "negative"]
            ]
            for sample in data
        ]
        for j in i
    ]
    offsets = [
        j
        for i in [
            [
                entity["offset"]
                for entity in sample["entities"]
                if sample["items"][str(entity["id"])]
                in ["positive", "neutral", "negative"]
            ]
            for sample in data
        ]
        for j in i
    ]
    labels = [
        j
        for i in [
            [
                {"positive": 1, "neutral": 0, "negative": -1}.get(
                    sample["items"][str(entity["id"])]
                )
                for entity in sample["entities"]
                if sample["items"][str(entity["id"])]
                in ["positive", "neutral", "negative"]
            ]
            for sample in data
        ]
        for j in i
    ]

    return sentences, targets, offsets, labels


def xue_parser(path):
    data = loads(open(path, "r").read())
    data = [sample for sample in data if sample["aspect"] != "conflict"]
    sentences = [
        sample["sentence"]
        for sample in data
        if sample["sentiment"] in ["positive", "neutral", "negative"]
    ]
    targets = [
        sample["aspect"]
        for sample in data
        if sample["sentiment"] in ["positive", "neutral", "negative"]
    ]
    labels = [
        {"positive": 1, "neutral": 0, "negative": -1}.get(sample["sentiment"])
        for sample in data
        if sample["sentiment"] in ["positive", "neutral", "negative"]
    ]
    return sentences, targets, labels
