from decimal import Decimal
import six
import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
from PIL import Image, ImageDraw, ImageFont, ImageOps
import matplotlib
from matplotlib.font_manager import FontProperties
from scipy.special import softmax
from tsaplay.constants import DEFAULT_FONT
from tsaplay.utils.io import get_image_from_plt, pickle_file, platform

if platform() == "MacOS":
    matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib_venn import venn2, venn2_unweighted, venn3, venn3_unweighted


def cmap_int(value, cmap_name="Oranges", alpha=0.8):
    cmap = plt.get_cmap(cmap_name)
    rgba_flt = cmap(value, alpha=alpha)
    rgba_arr = matplotlib.colors.to_rgba_array(rgba_flt)[0]
    rgba_int = np.int32(rgba_arr * 255)
    return tuple(rgba_int)


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
    joined_image_with_border = ImageOps.expand(
        joined_image, border=border, fill="black"
    )
    return joined_image_with_border


def draw_attention_heatmap(phrases, attn_vecs, **kwargs):
    font = ImageFont.truetype(font=DEFAULT_FONT, size=16)

    phrases = [[str(t, "utf-8") for t in p if t != b""] for p in phrases]
    attn_vecs = [
        np.transpose(normalize(np.transpose(attn_vec), norm="l2"))
        for attn_vec in attn_vecs
    ]
    attn_vecs = [a[: len(p)] for a, p in zip(attn_vecs, phrases)]

    v_space = 5
    h_space = 10
    images = []
    phrase_images = []
    full_phrase = " ".join(map(lambda phrase: " ".join(phrase), phrases))
    max_height = font.getsize(text=full_phrase)[1] + h_space
    for phrase, attn_vec in zip(phrases, attn_vecs):
        for token, attn_val in zip(phrase, attn_vec):
            color = cmap_int(attn_val[0], **kwargs)
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
    """
    attn_vecs: can have 2 different shape types
        with hops: (numpy ndarray) [HOP][PHRASE][WORD][VAL]
        without: 2d numpy array of list [0][PHRASE][list][list] 
    """
    images = []
    phrases = [[str(t, "utf-8") for t in p if t != b""] for p in phrases]
    has_hops = len(np.shape(attn_vecs)) == 4
    n_hops = len(attn_vecs)
    p_ordered_attn_vecs = (
        attn_vecs[0]
        if not has_hops
        else np.transpose(attn_vecs, axes=(1, 0, 2, 3))
    )
    for index, phrase in enumerate(phrases):
        if not phrase:
            continue
        cols = phrase
        if has_hops:
            cols = ["Hop"] + cols
            shading_data = np.squeeze(
                [
                    normalize(np.transpose(hop[: len(phrase)]))
                    for hop in p_ordered_attn_vecs[index]
                ]
            )
            shading_data = np.concatenate(
                (
                    np.expand_dims(np.array(range(1, n_hops + 1)), axis=1),
                    shading_data,
                ),
                axis=1,
            )
            data = np.squeeze(
                [
                    softmax(np.transpose(hop[: len(phrase)]))
                    if np.sum(hop) != 1
                    else np.transpose(hop[: len(phrase)])
                    for hop in p_ordered_attn_vecs[index]
                ]
            )
            data = np.concatenate(
                (np.expand_dims(np.array(range(1, n_hops + 1)), axis=1), data),
                axis=1,
            )
        else:
            shading_data = normalize(
                [[el[0] for el in p_ordered_attn_vecs[index][: len(phrase)]]]
            )
            data = [
                softmax(
                    [el[0] for el in p_ordered_attn_vecs[index][: len(phrase)]]
                )
            ]
        image = render_mpl_table(
            data=pd.DataFrame(data=data, columns=cols),
            shade_data=pd.DataFrame(data=shading_data, columns=cols),
        )
        images.append(image)
    return join_images(images, v_space=0, padding=0)


def draw_prediction_label(target, label, prediction, classes):
    h_space = 10
    v_space = 5
    font = ImageFont.truetype(font=DEFAULT_FONT, size=16)
    text = "Target: {0} \t\t Predicted: {1} \t Correct: {2}".format(
        target, classes[prediction], classes[label]
    )

    _, max_height = font.getsize(text)

    images = []
    words = text.split()
    for i, word in enumerate(words):
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
            text=word,
            fill=fill,
            font=font,
        )
        images.append(img)

    final_image = join_images(images)

    return final_image


def render_mpl_table(
    data,
    shade_data=None,
    col_width=1,
    row_height=0.3,
    header_color="#40466e",
    edge_color="w",
    bbox=[0, 0, 1, 1],
    ax=None,
    **kwargs
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
    shade_arr = np.asarray(shade_data) if shade_data is not None else data_arr
    n_hops = len(data_arr)
    alpha_inc = 1 / n_hops
    cell_text = [[pretty_print_decimal(i) for i in j] for j in data.values]

    mpl_table = ax.table(
        cellText=cell_text,
        bbox=bbox,
        colLabels=data.columns,
        cellLoc="center",
        **kwargs
    )

    font = FontProperties(fname=DEFAULT_FONT, size=12)
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
                col = matplotlib.colors.to_rgba(header_color, alpha=0.7)
                cell.set_facecolor(col)
        elif k[1] == 0 and n_hops > 1:
            cell.set_text_props(weight="bold", color="w")
            if k[0] == 0:
                cell.set_facecolor(header_color)
            else:
                col = matplotlib.colors.to_rgba(
                    header_color, alpha=(0 + alpha_inc * k[0])
                )
                cell.set_facecolor(col)
        else:
            shade_val = shade_arr[k[0] - 1, k[1]]
            cell.set_facecolor(cmap(shade_val, alpha=0.8))

    image = get_image_from_plt(plt)

    return image


def plot_distributions(stats, mode):
    outer_labels = np.array([[*stats]]).flatten()
    vals = np.array(
        [
            [float(_cls["count"]) for kc, _cls in ds[mode].items()]
            for k, ds in stats.items()
        ]
    )

    _, ax = plt.subplots(figsize=(18, 12))

    outer_cmap = plt.get_cmap("tab20c")
    outer_colors = outer_cmap((np.arange(len(stats)) * 4 + 1) % 16)
    inner_cmap = plt.get_cmap("tab20")
    inner_colors = inner_cmap([6, 0, 4])

    ax.pie(
        vals.sum(axis=1),
        radius=1,
        colors=outer_colors,
        labels=outer_labels,
        wedgeprops=dict(width=0.2, edgecolor="w"),
    )

    def func(pct, allvals):
        if np.ndim(allvals) > 1:
            return "{:.1f}%".format(pct)
        absolute = int(pct / 100.0 * np.sum(allvals))
        return "{:.1f}%\n({:d})".format(pct, absolute)

    wedges, _, autotexts = ax.pie(
        vals.flatten(),
        radius=0.8,
        colors=inner_colors,
        autopct=lambda pct: func(pct, vals),
        wedgeprops={"width": 0.45, "edgecolor": "w"},
        textprops={"color": "w"},
    )

    ax.legend(
        wedges,
        ["-1", "0", "1"],
        title="Classes",
        loc="center left",
        bbox_to_anchor=(1, 0, 0.5, 1),
    )

    plt.setp(autotexts, size=12, weight="bold")

    totals = np.sum(vals, axis=0)
    wedges, _, autotexts = ax.pie(
        totals,
        radius=0.35,
        colors=inner_colors,
        autopct=lambda pct: func(pct, totals),
        wedgeprops={"width": 0.35, "edgecolor": "w"},
        textprops={"color": "w"},
    )

    plt.setp(autotexts, size=12, weight="bold")

    ax.set_title(mode.capitalize() + " Data Distribution")

    return get_image_from_plt(plt)


def draw_venn(title, weighted=False, **kwargs):
    num_sets = len([*kwargs])
    if num_sets == 2:
        venn_fn = venn2 if weighted else venn2_unweighted
    elif num_sets == 3:
        venn_fn = venn3 if weighted else venn3_unweighted
    else:
        raise ValueError(
            "Number of sets {} must be either 2 or 3.".format(num_sets)
        )

    plt.figure(figsize=(4, 4))
    venn_fn(kwargs.values(), [*kwargs])
    plt.title(title)
    return get_image_from_plt(plt)
