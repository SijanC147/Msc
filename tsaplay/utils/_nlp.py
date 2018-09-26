import spacy
import numpy as np
import matplotlib as mpl
from PIL import Image, ImageDraw, ImageFont, ImageOps
from random import choice, randrange
from math import floor
from matplotlib.font_manager import FontProperties
from spacy.attrs import ORTH  # pylint: disable=E0611


mpl.use("TkAgg")
import matplotlib.pyplot as plt  # nopep8


def tokenize_phrase(phrase, lower=False):
    tokens_list = []
    nlp = spacy.load("en", disable=["parser", "ner"])
    tokens = nlp(str(phrase))
    tokens_list = list(filter(token_filter, tokens))
    return [
        token.text.lower() if lower else token.text for token in tokens_list
    ]


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
    mappings_zip = zip(
        features["mappings"]["left"],
        features["mappings"]["target"],
        features["mappings"]["right"],
    )
    lengths = [len(l + t + r) for (l, t, r) in mappings_zip]
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
        "mean_sen_length": round(np.mean(lengths), 2),
    }


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
        offset = sentence.lower().find(target.lower())
    left = sentence[:offset]
    start = offset + len(target)
    right = sentence[start:]
    return left.strip(), right.strip()


def get_sentence_target_features(
    embedding, sentence, target, label=None, offset=None
):
    sentence_literal = sentence.strip()
    target_literal = target.strip()

    left_literal, right_literal = get_sentence_contexts(
        sentence=sentence_literal, target=target_literal, offset=offset
    )

    left_mapping = embedding.get_index_ids(left_literal)
    target_mapping = embedding.get_index_ids(target_literal)
    right_mapping = embedding.get_index_ids(right_literal)

    sentence_length = len(left_mapping + target_mapping + right_mapping)

    if label is not None and isinstance(label, str):
        label = int(label.strip())

    return {
        "sentence": sentence_literal,
        "sentence_len": sentence_length,
        "left_lit": left_literal,
        "target_lit": target_literal,
        "right_lit": right_literal,
        "left_map": left_mapping,
        "target_map": target_mapping,
        "right_map": right_mapping,
        "label": label,
    }


def cmap_int(value, cmap_name="Oranges", alpha=0.8):
    cmap = plt.get_cmap(cmap_name)
    rgba_flt = cmap(value, alpha=alpha)
    rgba_arr = mpl.colors.to_rgba_array(rgba_flt)[0]
    rgba_int = np.int32(rgba_arr * 255)
    return tuple(rgba_int)


def draw_attention_heatmap(phrases, attn_vecs):
    font = ImageFont.truetype(font="./tsaplay/Symbola.ttf", size=24)

    phrases = [[t for t in tokenize_phrase(str(p, "utf-8"))] for p in phrases]
    attn_vecs = [a[: len(p)] for a, p in zip(attn_vecs, phrases)]

    # for tokens, vectors in zip(phrases, attn_vecs):
    #     print(" ".join(tokens))
    #     print(" ".join(str(np.squeeze(vectors, axis=1))))

    v_space = 5
    h_space = 10
    images = []
    phrase_images = []
    full_phrase = " ".join(map(lambda phrase: " ".join(phrase), phrases))
    max_height = font.getsize(text=full_phrase)[1] + h_space
    for phrase, attn_vec in zip(phrases, attn_vecs):
        for token, attn_val in zip(phrase, attn_vec):
            color = cmap_int(attn_val[0])
            size = font.getsize(text=token)
            img = Image.new(
                mode="RGBA", size=(size[0] + v_space, max_height), color=color
            )
            draw = ImageDraw.Draw(img)
            draw.text(
                xy=(int(v_space / 2), int(h_space / 2)),
                text=token,
                fill=(0, 0, 0),
                font=font,
            )
            images.append(img)
        if len(images) > 0:
            phrase_image_with_border = join_images(images=images, border=1)
            phrase_images.append(phrase_image_with_border)
        images = []

    new_image = join_images(phrase_images, v_space=15)

    return new_image


def get_class_text(class_id):
    return {0: "Negative", 1: "Neutral", 2: "Positive"}.get(class_id)


def draw_prediction_label(target, label, prediction):
    h_space = 10
    v_space = 5
    font = ImageFont.truetype(font="./tsaplay/Symbola.ttf", size=16)
    text = "Target: {0} \t\t Predicted: {1} \t Correct: {2}".format(
        target, get_class_text(prediction), get_class_text(label)
    )

    _, max_height = font.getsize(text)

    images = []
    words = text.split()
    for i in range(len(words)):
        width, _ = font.getsize(words[i])
        img = Image.new(
            mode="RGBA", size=(width + v_space, max_height + h_space)
        )
        draw = ImageDraw.Draw(img)
        if i > 0 and words[i - 1] == "Predicted:":
            if label != prediction:
                fill = (255, 0, 0)
            else:
                fill = (0, 255, 0)
        else:
            fill = (0, 0, 0)

        draw.text(
            xy=(int(v_space / 2), int(h_space / 2)),
            text=words[i],
            fill=fill,
            font=font,
        )
        images.append(img)

    final_image = join_images(images)

    return final_image


def stack_images(images, h_space=10):
    widths, heights = zip(*(im.size for im in images))

    total_height = sum(heights) + h_space * len(images)
    max_width = sum(widths)

    stacked_image = Image.new(mode="RGBA", size=(max_width, total_height))

    y_offset = 0

    for im in images:
        stacked_image.paste(im, (0, y_offset))
        y_offset += im.size[1] + h_space

    return stacked_image


def join_images(images, v_space=5, border=None, padding=2):
    widths, heights = zip(*(im.size for im in images))

    total_width = sum(widths) + v_space * (len(images) - 1) + 2 * padding
    max_height = max(heights) + 2 * padding

    joined_image = Image.new(mode="RGBA", size=(total_width, max_height))

    x_offset = padding

    for im in images:
        joined_image.paste(im, (x_offset, padding))
        x_offset += im.size[0] + v_space

    if border is None:
        return joined_image
    else:
        joined_image_with_border = ImageOps.expand(
            joined_image, border=border, fill="black"
        )
        return joined_image_with_border
