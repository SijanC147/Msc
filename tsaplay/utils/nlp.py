import spacy
import numpy as np
import matplotlib as mpl
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont, ImageOps
from random import choice, randrange
from math import floor
from matplotlib.font_manager import FontProperties
from spacy.attrs import ORTH  # pylint: disable=E0611


mpl.use("TkAgg")
import matplotlib.pyplot as plt  # nopep8


def tokenize_phrases(phrases):
    token_lists = []
    nlp = spacy.load("en", disable=["parser", "ner"])
    for doc in tqdm(nlp.pipe(phrases, batch_size=100, n_threads=-1)):
        tokens = list(filter(token_filter, doc))
        token_lists.append([t.text.lower() for t in tokens])
    return token_lists


def token_filter(token):
    if token.like_url:
        return False
    if token.like_email:
        return False
    if token.text in ["\uFE0F"]:
        return False
    return True


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


def inspect_dist(left, target, right, labels):
    positive = [label for label in labels if label == 1]
    neutral = [label for label in labels if label == 0]
    negative = [label for label in labels if label == -1]
    lengths = [
        len(np.concatenate([l, t, r])) for l, t, r in zip(left, target, right)
    ]
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


def cmap_int(value, cmap_name="Oranges", alpha=0.8):
    cmap = plt.get_cmap(cmap_name)
    rgba_flt = cmap(value, alpha=alpha)
    rgba_arr = mpl.colors.to_rgba_array(rgba_flt)[0]
    rgba_int = np.int32(rgba_arr * 255)
    return tuple(rgba_int)


def draw_attention_heatmap(phrases, attn_vecs):
    font = ImageFont.truetype(font="./tsaplay/Symbola.ttf", size=24)

    phrases = [[str(t, "utf-8") for t in p if t != b""] for p in phrases]
    attn_vecs = [a[: len(p)] for a, p in zip(attn_vecs, phrases)]

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


def draw_prediction_label(target, label, prediction, classes):
    h_space = 10
    v_space = 5
    font = ImageFont.truetype(font="./tsaplay/Symbola.ttf", size=16)
    text = "Target: {0} \t\t Predicted: {1} \t Correct: {2}".format(
        target, classes[prediction], classes[label]
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
