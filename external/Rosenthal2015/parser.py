from csv import DictReader


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
