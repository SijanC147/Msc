import re
from csv import DictReader


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
            sentence, target = row["sentence"], row["target"]
            if len(re.findall(r"\b{}\b"format(target), sentence, re.IGNORECASE)) > 1:
                continue
            sentences.append(sentence)
            targets.append(target)
            labels.append(
                {"2": 1, "1": 1, "0": 0, "-1": -1, "-2": -1}.get(
                    row["sentiment"]
                )
            )
    return sentences, targets, labels
