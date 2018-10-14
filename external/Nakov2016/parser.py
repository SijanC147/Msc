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
            sentences.append(row["sentence"])
            targets.append(row["target"])
            labels.append(
                {"2": 1, "1": 1, "0": 0, "-1": -1, "-2": -1}.get(
                    row["sentiment"]
                )
            )
    return sentences, targets, labels
