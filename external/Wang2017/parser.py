from json import loads
from os.path import join, dirname
from itertools import chain
from collections import defaultdict


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
