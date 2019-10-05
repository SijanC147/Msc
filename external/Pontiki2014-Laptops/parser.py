from xml.dom import minidom


def pontiki_parser(path):
    sentences = []
    targets = []
    labels = []
    offsets = []
    xmldata = minidom.parse(path)
    for sentence in xmldata.getElementsByTagName("sentences")[
        0
    ].getElementsByTagName("sentence"):
        try:
            for term in sentence.getElementsByTagName("aspectTerms")[
                0
            ].getElementsByTagName("aspectTerm"):
                polarity = {"positive": 1, "neutral": 0, "negative": -1}.get(
                    term.attributes["polarity"].value
                )
                if polarity is not None:
                    sentences.append(
                        sentence.getElementsByTagName("text")[
                            0
                        ].firstChild.data
                    )
                    targets.append(term.attributes["term"].value)
                    offsets.append(int(term.attributes["from"].value))
                    labels.append(polarity)
        except IndexError:
            continue
    return sentences, targets, offsets, labels
