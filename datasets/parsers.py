def dong_parser(file):
    lines = open(file, "r").readlines()
    sentences = [
        lines[index].strip().replace("$T$", lines[index + 1].strip())
        for index in range(0, len(lines), 3)
    ]
    targets = [lines[index + 1].strip() for index in range(0, len(lines), 3)]
    labels = [
        int(lines[index + 2].strip()) for index in range(0, len(lines), 3)
    ]
    return sentences, targets, labels
