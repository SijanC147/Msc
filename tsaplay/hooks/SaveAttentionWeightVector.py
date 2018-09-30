import tensorflow as tf
import numpy as np

import textwrap
import re
import io
import itertools
import matplotlib

from PIL import Image, ImageDraw
from tensorflow.train import SessionRunHook, SessionRunArgs
from tsaplay.utils.nlp import (
    draw_attention_heatmap,
    draw_prediction_label,
    stack_images,
    tokenize_phrase,
)
from tsaplay.utils.tf import image_to_summary

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt  # nopep8


class SaveAttentionWeightVector(SessionRunHook):
    def __init__(
        self,
        labels,
        predictions,
        targets,
        summary_writer,
        n_picks=3,
        n_hops=None,
    ):
        self.labels = labels
        self.predictions = predictions
        self.targets = targets
        self.n_picks = n_picks
        self.n_hops = n_hops
        self._summary_writer = summary_writer

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
        if self.n_picks is None:
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
            else:
                i_hop = i * self.n_hops
                hop_images = []
                for ih in range(i_hop, i_hop + self.n_hops):
                    phrases = [
                        attn_mech[0][ih].tolist() for attn_mech in attn_mechs
                    ]
                    attn_vecs = [
                        attn_mech[1][ih].tolist() for attn_mech in attn_mechs
                    ]
                    attn_heatmap_hop = draw_attention_heatmap(
                        phrases=phrases, attn_vecs=attn_vecs
                    )
                    hop_images.append(attn_heatmap_hop)
                attn_heatmap = stack_images(hop_images, h_space=10)

            target = " ".join(
                [str(t, "utf-8") for t in targets[i] if t != b""]
            )
            label = labels[i]
            prediction = preds[i]
            pred_label = draw_prediction_label(
                target=target, label=label, prediction=prediction
            )

            images.append(stack_images([attn_heatmap, pred_label], h_space=10))

        final_image = stack_images(images, h_space=40)

        summary = image_to_summary(
            name="Attention Heatmaps", image=final_image
        )

        self._summary_writer.add_summary(summary, global_step)

    def _tile_over_hops(self, value, n_hops):
        value = np.expand_dims(value, axis=1)
        value = np.tile(value, reps=[1, n_hops])
        value = np.reshape(value, newshape=[-1])

        return value
