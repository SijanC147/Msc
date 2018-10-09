import spacy
import six
import pandas as pd
import numpy as np
import matplotlib as mpl
from decimal import Decimal
from io import BytesIO
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont, ImageOps
from random import choice, randrange
from math import floor
from matplotlib.font_manager import FontProperties
from spacy.attrs import ORTH  # pylint: disable=E0611
from tsaplay.utils.io import cprnt, get_image_from_plt


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


def cmap_int(value, cmap_name="Oranges", alpha=0.8):
    cmap = plt.get_cmap(cmap_name)
    rgba_flt = cmap(value, alpha=alpha)
    rgba_arr = mpl.colors.to_rgba_array(rgba_flt)[0]
    rgba_int = np.int32(rgba_arr * 255)
    return tuple(rgba_int)


def draw_attention_heatmap(phrases, attn_vecs):
    font = ImageFont.truetype(font="./tsaplay/Symbola.ttf", size=16)

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


def tabulate_attention_value(phrases, attn_vecs):
    images = []
    phrases = [[str(t, "utf-8") for t in p if t != b""] for p in phrases]
    n_hops = len(attn_vecs)
    for index, phrase in enumerate(phrases):
        if not phrase:
            continue
        df = pd.DataFrame()
        if n_hops > 1:
            df["Hop"] = [h + 1 for h in range(n_hops)]
        for w_index, word in enumerate(phrase):
            df[word] = np.concatenate(
                [attn_vecs[h][index][w_index] for h in range(len(attn_vecs))]
            )
        images.append(render_mpl_table(df))
    return join_images(images, v_space=0, padding=0)


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


def render_mpl_table(
    data,
    col_width=1,
    row_height=0.3,
    header_color="#40466e",
    row_colors=["#f1f1f2", "w"],
    edge_color="w",
    bbox=[0, 0, 1, 1],
    ax=None,
    **kwargs,
):
    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array(
            [col_width, row_height]
        )
        fig, ax = plt.subplots(figsize=size)
        ax.axis("off")

    def pretty_print_decimal(value, precision=4):
        str_value = ("{:." + str(precision) + "f}").format(value)
        decimal = Decimal(str_value).normalize()
        return str(decimal)

    data_arr = np.asarray(data)
    n_hops = len(data_arr)
    alpha_inc = 1 / n_hops
    cell_text = [[pretty_print_decimal(i) for i in j] for j in data.values]

    mpl_table = ax.table(
        cellText=cell_text,
        bbox=bbox,
        colLabels=data.columns,
        cellLoc="center",
        **kwargs,
    )

    font = FontProperties(fname="./tsaplay/Symbola.ttf", size=12)
    mpl_table.auto_set_font_size(False)

    cmap = plt.get_cmap("Oranges")
    for k, cell in six.iteritems(mpl_table._cells):
        cell.set_edgecolor(edge_color)
        cell.set_text_props(fontproperties=font)
        if k[0] == 0:
            if k[1] == 0 and n_hops > 1:
                cell.set_text_props(weight="bold", color="w")
                cell.set_facecolor(header_color)
            else:
                cell.set_text_props(color="w")
                col = mpl.colors.to_rgba(header_color, alpha=0.7)
                cell.set_facecolor(col)
        elif k[1] == 0 and n_hops > 1:
            cell.set_text_props(weight="bold", color="w")
            if k[0] == 0:
                cell.set_facecolor(header_color)
            else:
                col = mpl.colors.to_rgba(
                    header_color, alpha=(0 + alpha_inc * k[0])
                )
                cell.set_facecolor(col)
        else:
            cell.set_facecolor(cmap(data_arr[k[0] - 1, k[1]], alpha=0.8))

    image = get_image_from_plt(plt)

    return image


def stack_images(images, h_space=10):
    if not images:
        return
    widths, heights = zip(*(im.size for im in images))

    total_height = sum(heights) + h_space * len(images)
    max_width = max(widths)

    stacked_image = Image.new(mode="RGBA", size=(max_width, total_height))

    y_offset = 0

    for im in images:
        stacked_image.paste(im, (0, y_offset))
        y_offset += im.size[1] + h_space

    return stacked_image


def join_images(images, v_space=5, border=None, padding=2):
    if not images:
        return
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


def re_dist(labels, target_dist):
    counts = [
        len([l for l in labels if l == -1]),
        len([l for l in labels if l == 0]),
        len([l for l in labels if l == 1]),
    ]
    target_counts = [0, 0, 0]

    if 1 in target_dist:
        total_index = target_dist.index(1)
        target_counts[total_index] = counts[total_index]
        return target_counts

    smallest_count_indices = [
        i for i, x in enumerate(counts) if x == min(counts)
    ]
    if len(smallest_count_indices) != 1:
        smallest_count_index = target_dist.index(
            max([target_dist[i] for i in smallest_count_indices])
        )
    else:
        smallest_count_index = smallest_count_indices[0]

    target_counts[smallest_count_index] = counts[smallest_count_index]
    new_total = floor(
        counts[smallest_count_index] / target_dist[smallest_count_index]
    )
    counts[smallest_count_index] = float("inf")
    target_dist[smallest_count_index] = float("inf")

    smallest_count_indices = [
        i for i, x in enumerate(counts) if x == min(counts)
    ]
    if len(smallest_count_indices) != 1:
        smallest_count_index = target_dist.index(
            max([target_dist[i] for i in smallest_count_indices])
        )
    else:
        smallest_count_index = smallest_count_indices[0]
    smallest_count_index = counts.index(min(counts))
    target_count = int(new_total * target_dist[smallest_count_index])
    if target_count > counts[smallest_count_index]:
        old_total = new_total
        new_total = floor(
            counts[smallest_count_index] / target_dist[smallest_count_index]
        )
        target_counts = [
            floor(t * (new_total / old_total)) for t in target_counts
        ]
        target_count = counts[smallest_count_index]
    target_counts[smallest_count_index] = target_count
    counts[smallest_count_index] = float("inf")
    target_dist[smallest_count_index] = float("inf")

    if new_total - sum(target_counts) > min(counts):
        old_total = new_total
        new_total = floor(min(counts) / min(target_dist))
        target_counts = [
            floor(t * (new_total / old_total)) for t in target_counts
        ]
    target_counts[target_counts.index(0)] = new_total - sum(target_counts)

