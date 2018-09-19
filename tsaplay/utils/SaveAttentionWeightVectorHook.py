import tensorflow as tf
import numpy as np

import textwrap
import re
import io
import itertools
import matplotlib

from PIL import Image, ImageDraw
from tensorflow.train import SessionRunHook, SessionRunArgs
from tsaplay.utils._nlp import (
    draw_attention_heatmap,
    draw_prediction_label,
    stack_images,
)
from tsaplay.utils._tf import image_to_summary

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt  # nopep8


class SaveAttentionWeightVectorHook(SessionRunHook):
    def __init__(self, labels, predictions, targets, summary_writer, picks=1):
        self.labels = labels
        self.predictions = predictions
        self.targets = targets
        self.picks = picks
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
        results = run_values.results
        attn_mechs = results["attention"]
        targets = results["targets"]
        labels = results["labels"]
        predictions = results["predictions"]
        num_attn_units = len(attn_mechs)
        if num_attn_units < 1:
            return
        global_step = results["global_step"][0]
        attn_samples = attn_mechs[0][0]
        if self.picks > len(attn_samples):
            self.picks = len(attn_samples)
        rnd_picked = np.random.choice(
            attn_samples, size=self.picks, replace=False
        )
        indices = [attn_samples.tolist().index(pick) for pick in rnd_picked]

        images = []
        for i in indices:
            phrases = [attn_mech[0][i] for attn_mech in attn_mechs]
            attn_vecs = [attn_mech[1][i] for attn_mech in attn_mechs]
            target = str(targets[i], "utf-8")
            label = labels[i]
            prediction = predictions[i]

            attn_heatmap = draw_attention_heatmap(
                phrases=phrases, attn_vecs=attn_vecs
            )
            pred_label = draw_prediction_label(
                target=target, label=label, prediction=prediction
            )
            images.append(stack_images([attn_heatmap, pred_label], h_space=10))

        final_image = stack_images(images, h_space=40)

        summary = image_to_summary(
            name="Attention Heatmaps", image=final_image
        )

        self._summary_writer.add_summary(summary, global_step)
