from json import loads


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
