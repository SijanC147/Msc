import textwrap
import re
import itertools
import os
import traceback
from os import path
from math import ceil
from warnings import warn
from datetime import datetime
import time
import matplotlib
import numpy as np
import tensorflow as tf
from tensorflow.train import (
    SessionRunHook,
    SessionRunArgs,
    CheckpointSaverListener,
)  # noqa
from tensorflow.estimator import ModeKeys  # noqa
from tsaplay.constants import NP_RANDOM_SEED
from tsaplay.utils.draw import (
    draw_attention_heatmap,
    draw_prediction_label,
    stack_images,
    tabulate_attention_value,
)
from tsaplay.utils.tf import image_to_summary, checkpoints_state_data
from tsaplay.utils.io import (
    temp_pngs,
    get_image_from_plt,
    pickle_file,
    cprnt,
    platform,
    search_dir,
)
from tsaplay.utils.debug import timeit

if platform() == "MacOS":
    matplotlib.use("TkAgg")
import matplotlib.pyplot as plt  # noqa pylint: disable=C0411,C0412,C0413


class DiscardRedundantStopSignalCheckpoint(CheckpointSaverListener):
    def __init__(self, model_dir, comet=None):
        self.model_dir = model_dir
        self.comet = comet

    # pylint: disable=unused-argument
    def after_save(self, session, global_step_value):
        stopped_early = (
            tf.get_default_graph()
            .get_tensor_by_name("signal_early_stopping/STOP:0")
            .eval(session=session)
            .astype(bool)
        )
        # cprnt(
        #     train="Discard Chckpt-{}? {}".format(
        #         global_step_value, "YEP" if stopped_early else "NOPE"
        #     )
        # )
        if stopped_early:
            dir_str, query_str = path.split(
                tf.train.latest_checkpoint(self.model_dir)
            )
            if dir_str.startswith("gs://"):
                file_paths = [
                    path.join(dir_str, fname)
                    for fname in tf.gfile.ListDirectory(dir_str)
                    if fname.startswith(query_str)
                ]
                for old_file_path in file_paths:
                    tf.gfile.Remove(old_file_path)
                    cprnt(tf=True, warn="DEL: {}".format(old_file_path))
            else:
                for old_file_path in search_dir(dir_str, query=query_str):
                    os.remove(old_file_path)
                    cprnt(tf=True, warn="DEL: {}".format(old_file_path))
            if self.comet is not None:
                _ts = datetime.timestamp(datetime.now())
                self.comet.log_other("END", str(_ts))
                self.comet.disable_mp()
                self.comet.end()
            summary_writer = tf.summary.FileWriterCache.get(self.model_dir)
            summary_writer.flush()


class SummarySavingHook(SessionRunHook):
    def __init__(self, ops, every_n_iter, writer, first_step):
        self.summary_op = ops
        self.summary_freq = every_n_iter
        self._summary_writer = writer
        self._global_step = first_step

    def before_run(self, _):
        # cprnt(train="BEFORE Summary Saving ({})".format(self._global_step))
        this_step = self._global_step + 1
        fetches = {
            "global_step": tf.get_collection(tf.GraphKeys.GLOBAL_STEP),
            **(
                {"summary": self.summary_op}
                if this_step % self.summary_freq == 0 or this_step == 1
                else {}
            ),
        }
        return SessionRunArgs(fetches=fetches)

    def after_run(self, _, run_values):
        # cprnt(train="AFTER Summary Saving ({})".format(self._global_step))
        global_step = run_values.results.pop("global_step")[0]
        self._global_step = global_step
        if "summary" in [*run_values.results] and (
            global_step % self.summary_freq == 0 or global_step == 1
        ):
            summary_data = run_values.results.pop("summary")
            self._summary_writer.add_summary(summary_data, global_step)
            cprnt(
                tf=True, info="Saved summary for step {}".format(global_step)
            )

    def end(self, _):
        # cprnt(train="END Summary Saving ({})".format(self._global_step))
        self._summary_writer.flush()


class ConsoleLoggerHook(SessionRunHook):
    # pylint: disable=too-many-arguments
    def __init__(
        self, mode, tensors, template, every_n_iter=None, epoch_steps=None
    ):
        self.mode = mode
        self.tensors = tensors
        self.template = template
        self.every_n_iter = every_n_iter or epoch_steps
        self.epoch_steps = epoch_steps
        self._start_time = None

    def begin(self):
        self._start_time = time.time()

    def before_run(self, _):
        # cprnt(**{self.mode: "BEFORE Console Logger"})
        fetches = {
            "global_step": tf.get_collection(tf.GraphKeys.GLOBAL_STEP),
            **(self.tensors if self.mode == ModeKeys.TRAIN else {}),
        }
        return SessionRunArgs(fetches=fetches)

    def after_run(self, _, run_values):
        # cprnt(**{self.mode: "AFTER Console Logger"})
        global_step = run_values.results.pop("global_step")[0]
        if self.mode == ModeKeys.TRAIN:
            if global_step % self.every_n_iter == 0 or global_step == 1:
                current_time = time.time()
                duration = current_time - self._start_time
                sec_per_step = float(
                    duration / (self.every_n_iter if global_step != 1 else 1)
                )
                step_per_sec = float(
                    (self.every_n_iter if global_step != 1 else 1) / duration
                )
                self._start_time = time.time()
                cprnt(
                    tf=True,
                    TRAIN=self.template.format_map(
                        {
                            "duration": duration,
                            "sec_per_step": sec_per_step,
                            "step_per_sec": step_per_sec,
                            "step": global_step,
                            "epoch": global_step / self.epoch_steps,
                            **run_values.results,
                        }
                    ),
                )

    def end(self, session):
        # cprnt(**{self.mode: "END Console Logger"})
        if self.mode == ModeKeys.EVAL:
            run_values = session.run(
                {"step": tf.train.get_global_step(), **self.tensors}
            )
            if self.epoch_steps is not None:
                cprnt(
                    tf=True,
                    EVAL=self.template.format_map(
                        {
                            **run_values,
                            **{
                                "epoch": run_values.get("step")
                                / self.epoch_steps
                            },
                        }
                    ),
                )


class LogProgressToComet(SessionRunHook):
    def __init__(self, mode, comet, epochs=None, epoch_steps=None):
        self.mode = mode
        self.comet = comet
        self.epoch_steps = epoch_steps
        self.epochs = epochs

    def begin(self):
        if self.mode == ModeKeys.TRAIN:
            _ts = datetime.timestamp(datetime.now())
            self.comet.log_other("START", str(_ts))

    def before_run(self, _):
        # cprnt(**{self.mode: "BEFORE Comet Progress"})
        return SessionRunArgs(
            fetches={
                "global_step": tf.get_collection(tf.GraphKeys.GLOBAL_STEP)
            }
        )

    def after_run(self, _, run_values):
        # cprnt(**{self.mode: "AFTER Comet Progress"})
        global_step = run_values.results["global_step"][0]
        self.comet.set_step(global_step)
        if self.epoch_steps:
            epoch = ceil(global_step / self.epoch_steps)
            self.comet.set_epoch(epoch)
            if self.mode == ModeKeys.TRAIN:
                if global_step % self.epoch_steps == 0:
                    self.comet.log_epoch_end(epoch)

    def end(self, session):
        # cprnt(**{self.mode: "END Comet Progress"})
        global_step = session.run(tf.train.get_global_step())
        if self.epoch_steps:
            epoch = ceil(global_step / self.epoch_steps)
            self.comet.log_metric("epoch", epoch, include_context=False)
            if global_step % self.epoch_steps == 0:
                self.comet.log_epoch_end(epoch)
        if self.mode == ModeKeys.TRAIN:
            _ts = datetime.timestamp(datetime.now())
            self.comet.log_other("END", str(_ts))
            self.comet.end()
            # cprnt(info="Experiment Ended")


class LogHistogramsToComet(SessionRunHook):
    def __init__(self, comet, names, trainables, every_n_iter):
        self.comet = comet
        self.names = names
        self.trainables = trainables
        self.every_n_iter = every_n_iter

    def before_run(self, _):
        # cprnt(train="BEFORE Comet Histograms")
        return SessionRunArgs(
            fetches={
                "global_step": tf.get_collection(tf.GraphKeys.GLOBAL_STEP),
                "trainables": self.trainables,
            }
        )

    def after_run(self, _, run_values):
        # cprnt(train="AFTER Comet Histograms")
        global_step = run_values.results["global_step"][0]
        if global_step % self.every_n_iter == 0 or global_step == 1:
            trainables = run_values.results["trainables"]
            for (name, trainable) in zip(self.names, trainables):
                self.comet.log_histogram_3d(
                    trainable, name=name, step=global_step
                )

    def end(self, session):
        # cprnt(train="END Comet Histograms")
        global_step = session.run(tf.train.get_global_step())
        trainables = session.run(self.trainables)
        for (name, trainable) in zip(self.names, trainables):
            self.comet.log_histogram_3d(trainable, name=name, step=global_step)


class MetadataHook(SessionRunHook):
    def __init__(self, mode, summary_writer, every_n_iter, first_step=None):
        self.mode = mode
        self.freq = every_n_iter
        self.counter = first_step or 0
        self._summary_writer = summary_writer

    def before_run(self, _):
        # cprnt(**{self.mode: "BEFORE Metadata"})
        this_step = self.counter + 1
        log_metadata = (
            self.mode == ModeKeys.EVAL and self.freq == "once"
            if isinstance(self.freq, str)
            else this_step == 1 or this_step % self.freq == 0
        )
        return SessionRunArgs(
            fetches={
                "global_step": tf.get_collection(tf.GraphKeys.GLOBAL_STEP)
            },
            options=(
                tf.RunOptions(
                    trace_level=tf.RunOptions.FULL_TRACE  # pylint: disable=E1101
                )
                if log_metadata
                else None
            ),
        )

    def after_run(self, _, run_values):
        # cprnt(**{self.mode: "AFTER Metadata"})
        if self.mode == ModeKeys.TRAIN:
            _counter = run_values.results["global_step"][0]
            tag = "step-{}".format(_counter)
        elif self.mode == ModeKeys.EVAL:
            if self.freq == "done":
                return
            eval_run = run_values.results["global_step"][0]
            self.counter += 1
            _counter = self.counter
            tag = "step-{}-batch-{}".format(eval_run, _counter)
        if self.freq == "once" or _counter == 1 or _counter % self.freq == 0:
            if self.freq == "once":
                self.freq = "done"
            try:
                self._summary_writer.add_run_metadata(
                    run_values.run_metadata, tag
                )
                cprnt(
                    tf=True, **{self.mode: "Logged Metadata ({})".format(tag)}
                )
            except ValueError:
                cprnt(
                    tf=True,
                    warn="{} Metadata ERR ({}) \n {}".format(
                        self.mode.upper(), tag, traceback.format_exc()
                    ),
                )

    def end(self, _):
        # cprnt(**{self.mode: "END Metadata"})
        self._summary_writer.flush()


class SaveAttentionWeightVector(SessionRunHook):
    def __init__(
        self,
        labels,
        predictions,
        targets,
        classes,
        summary_writer,
        comet=None,
        n_picks=1,
        n_hops=None,
        freq=5,
        epoch_steps=None,
    ):
        self.labels = labels
        self.predictions = predictions
        self.targets = targets
        self.classes = classes
        self.n_picks = n_picks
        self.n_hops = n_hops
        self.epoch_steps = epoch_steps
        self._summary_writer = summary_writer
        self._comet = comet
        self._counter = 0
        self.freq = freq

    def before_run(self, _):
        return SessionRunArgs(  # noqa
            fetches={
                "global_step": tf.get_collection(tf.GraphKeys.GLOBAL_STEP),
                "attention": tf.get_collection("ATTENTION"),
                "targets": self.targets,
                "labels": self.labels,
                "predictions": self.predictions,
            }
        )

    def after_run(self, _, run_values):
        if NP_RANDOM_SEED is not None:
            np.random.seed(NP_RANDOM_SEED)
        if self.n_picks is None:
            return
        pick = np.random.random()  # pylint: disable=no-member
        if isinstance(self.freq, float) and pick > self.freq:
            return
        if isinstance(self.freq, int) and self.freq <= 0:
            return
        if isinstance(self.freq, int):
            self.freq = self.freq - 1
        self._counter += 1
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
            epoch = (
                (global_step / self.epoch_steps) if self.epoch_steps else None
            )
            image_names = [
                (
                    "E{0:02.0f}#{1:02.0f} ".format(epoch, self._counter)
                    if epoch
                    else ""
                )
                + name
                for name in image_names
            ]
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
    def __init__(
        self,
        class_labels,
        tensor_name,
        summary_writer,
        comet=None,
        epoch_steps=None,
    ):
        self.tensor_name = tensor_name
        self.class_labels = class_labels
        self.epoch_steps = epoch_steps
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
            epoch = (
                (global_step / self.epoch_steps) if self.epoch_steps else None
            )

            title = (
                "{:.0f}. confusion_matrix".format(epoch)
                if epoch
                else "confusion_matrix"
            )
            for temp_png in temp_pngs([image], [title]):
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
        # fig.set_tight_layout(True)

        image = get_image_from_plt(plt)
        return image

