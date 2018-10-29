import textwrap
import re
import itertools
import numpy as np
import tensorflow as tf
from warnings import warn
from tensorflow.train import SessionRunHook, SessionRunArgs
import matplotlib
from tsaplay.constants import RANDOM_SEED
from tsaplay.utils.draw import (
    draw_attention_heatmap,
    draw_prediction_label,
    stack_images,
    tabulate_attention_value,
)
from tsaplay.utils.tf import image_to_summary
from tsaplay.utils.io import cprnt, temp_pngs, get_image_from_plt


# matplotlib.use("TkAgg")
import matplotlib.pyplot as plt  # noqa pylint: disable=C0411,C0412,C0413


class SaveAttentionWeightVector(SessionRunHook):
    def __init__(
        self,
        labels,
        predictions,
        targets,
        classes,
        summary_writer,
        comet=None,
        n_picks=3,
        n_hops=None,
        freq=0.2,
    ):
        self.labels = labels
        self.predictions = predictions
        self.targets = targets
        self.classes = classes
        self.n_picks = n_picks
        self.n_hops = n_hops
        self._summary_writer = summary_writer
        self._comet = comet
        self.freq = freq

    def before_run(self, run_context):
        return SessionRunArgs(
            fetches={
                "global_step": tf.get_collection(tf.GraphKeys.GLOBAL_STEP),
                "attention": tf.get_collection("ATTENTION"),
                "targets": self.targets,
                "labels": self.labels,
                "predictions": self.predictions,
            }
        )

    def after_run(self, run_context, run_values):
        np.random.seed(RANDOM_SEED)
        if self.n_picks is None or np.random.random() > self.freq:
            return
        results = run_values.results
        global_step = results["global_step"][0]
        attn_mechs = results["attention"]
        num_attn_units = len(attn_mechs)
        if num_attn_units < 1:
            return
        targets = results["targets"]
        labels = results["labels"]
        preds = results["predictions"]
        if self.n_picks > len(targets):
            self.n_picks = len(targets)
        indices = np.random.choice(
            list(range(len(targets))), size=self.n_picks, replace=False
        )

        images = []
        tables = []
        for i in indices:
            if self.n_hops is None:
                phrases = [
                    attn_mech[0][i].tolist() for attn_mech in attn_mechs
                ]
                attn_vecs = [
                    attn_mech[1][i].tolist() for attn_mech in attn_mechs
                ]
                attn_heatmap = draw_attention_heatmap(
                    phrases=phrases, attn_vecs=attn_vecs
                )
                attn_table = tabulate_attention_value(phrases, [attn_vecs])
            else:
                i_hop = i * self.n_hops
                hop_images = []
                hop_attns = []
                for hop_index in range(i_hop, i_hop + self.n_hops):
                    phrases = [
                        attn_mech[0][hop_index].tolist()
                        for attn_mech in attn_mechs
                    ]
                    attn_vecs = [
                        attn_mech[1][hop_index].tolist()
                        for attn_mech in attn_mechs
                    ]
                    attn_heatmap_hop = draw_attention_heatmap(
                        phrases=phrases, attn_vecs=attn_vecs
                    )
                    hop_images.append(attn_heatmap_hop)
                    hop_attns.append(attn_vecs)
                attn_table = tabulate_attention_value(phrases, hop_attns)
                attn_heatmap = stack_images(hop_images, h_space=10)

            target = " ".join(
                [str(t, "utf-8") for t in targets[i] if t != b""]
            )
            label = labels[i]
            prediction = preds[i]
            pred_label = draw_prediction_label(
                target=target,
                label=label,
                prediction=prediction,
                classes=self.classes,
            )

            images.append(stack_images([attn_heatmap, pred_label], h_space=10))
            tables.append(stack_images([attn_table, pred_label], h_space=0))

        final_heatmaps = stack_images(images, h_space=40)
        final_tables = stack_images(tables, h_space=40)

        if self._comet is not None:
            self._comet.set_step(global_step)
            image_names = ["attention_heatmap", "attention_tables"]
            images = [final_heatmaps, final_tables]
            for temp_png in temp_pngs(images, image_names):
                self._comet.log_image(temp_png)

        heatmap_summary = image_to_summary(
            name="Attention Heatmaps", image=final_heatmaps
        )
        self._summary_writer.add_summary(heatmap_summary, global_step)

        table_summary = image_to_summary(
            name="Attention Tables", image=final_tables
        )
        self._summary_writer.add_summary(table_summary, global_step)


class SaveConfusionMatrix(SessionRunHook):
    def __init__(self, class_labels, tensor_name, summary_writer, comet=None):
        self.tensor_name = tensor_name
        self.class_labels = class_labels
        self._summary_writer = summary_writer
        self._comet = comet

    def end(self, session):
        confusion_matrix = (
            tf.get_default_graph()
            .get_tensor_by_name(self.tensor_name + ":0")
            .eval(session=session)
            .astype(int)
        )
        global_step = tf.train.get_global_step().eval(session=session)
        image = self._plot_confusion_matrix(confusion_matrix)
        if self._comet is not None:
            for temp_png in temp_pngs([image], ["confusion_matrix"]):
                self._comet.set_step(global_step)
                self._comet.log_image(temp_png)

        summary = image_to_summary(name="Confusion Matrix", image=image)
        self._summary_writer.add_summary(summary, global_step)

    def _plot_confusion_matrix(self, conf_matrix):
        num_classes = len(self.class_labels)

        fig = plt.figure(
            figsize=(num_classes, num_classes), facecolor="w", edgecolor="k"
        )
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(conf_matrix, cmap="Oranges")

        classes = [
            re.sub(r"([a-z](?=[A-Z])|[A-Z](?=[A-Z][a-z]))", r"\1 ", x)
            for x in self.class_labels
        ]
        classes = ["\n".join(textwrap.wrap(l, 20)) for l in classes]

        tick_marks = np.arange(len(classes))

        ax.set_xlabel("Predicted")
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(classes, rotation=-90, ha="center")
        ax.xaxis.set_label_position("bottom")
        ax.xaxis.tick_bottom()

        ax.set_ylabel("True Label")
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(classes, va="center")
        ax.yaxis.set_label_position("left")
        ax.yaxis.tick_left()

        for i, j in itertools.product(range(num_classes), range(num_classes)):
            ax.text(
                j,
                i,
                int(conf_matrix[i, j]) if conf_matrix[i, j] != 0 else ".",
                horizontalalignment="center",
                verticalalignment="center",
                color="black",
            )
        fig.set_tight_layout(True)

        image = get_image_from_plt(plt)
        return image


class MetadataHook(SessionRunHook):
    def __init__(self, summary_writer, save_steps=None, save_batches=None):
        self._writer = summary_writer
        self._eval = save_batches is not None
        self._freq = save_steps or save_batches
        self._tag = "step-{0}" if not self._eval else "{0}-batch-{1}"

    def begin(self):
        if self._freq == "once":
            self._request_summary = True
        self._counter = 0 if not self._eval else 1
        self._global_step_tensor = tf.train.get_global_step()

        if self._global_step_tensor is None:
            raise RuntimeError(
                "Global step should be created to use ProfilerHook."
            )

    def before_run(self, run_context):
        requests = {"global_step": self._global_step_tensor}
        if self._freq != "once":
            self._request_summary = self._counter % self._freq == 0
        options = (
            tf.RunOptions(
                trace_level=tf.RunOptions.FULL_TRACE  # pylint: disable=E1101
            )
            if self._request_summary
            else None
        )

        return SessionRunArgs(requests, options=options)

    def after_run(self, run_context, run_values):
        global_step = run_values.results["global_step"]
        if self._request_summary:
            self._request_summary = False
            _id = [global_step] + ([self._counter] if self._eval else [])
            tag = self._tag.format(*_id)
            try:
                self._writer.add_run_metadata(run_values.run_metadata, tag)
            except ValueError:
                warn("Skipped metadata with tag {}.".format(tag))
        self._counter = (self._counter if self._eval else global_step) + 1
