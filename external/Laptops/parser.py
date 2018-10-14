from json import loads


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
