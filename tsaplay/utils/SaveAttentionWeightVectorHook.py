import tensorflow as tf
import numpy as np

import textwrap
import re
import io
import itertools
import matplotlib

from tensorflow.train import SessionRunHook, SessionRunArgs
from tsaplay.utils._nlp import draw_attention_heatmap
from tsaplay.utils._tf import figure_to_summary

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt  # nopep8


class SaveAttentionWeightVectorHook(SessionRunHook):
    def __init__(
        self,
        left_ctxts,
        targets,
        right_ctxts,
        labels,
        predictions,
        summary_writer,
    ):
        self.left_ctxts = left_ctxts
        self.targets = targets
        self.right_ctxts = right_ctxts
        self.labels = labels
        self.predictions = predictions
        self._summary_writer = summary_writer

    def before_run(self, run_context):
        return SessionRunArgs(
            fetches={
                "global_step": tf.get_collection(tf.GraphKeys.GLOBAL_STEP),
                "attn_vecs": tf.get_collection("ATTENTION_VECTORS"),
                "left_ctxts": self.left_ctxts,
                "targets": self.targets,
                "right_ctxts": self.right_ctxts,
                "labels": self.labels,
                "predictions": self.predictions,
            }
        )

    def after_run(self, run_context, run_values):
        results = run_values.results
        global_step = results["global_step"][0]
        random_chosen_targets = np.random.choice(results["targets"], 1)
        for trg in random_chosen_targets:
            index = results["targets"].tolist().index(trg)
            phrases = []
            attn_vec = []
            phrases.append(results["left_ctxts"][index])
            attn_vec.append(results["attn_vecs"][0][index])
            phrases.append(trg)
            attn_vec.append(results["attn_vecs"][2][index])
            phrases.append(trg)
            attn_vec.append(results["attn_vecs"][3][index])
            phrases.append(results["right_ctxts"][index])
            attn_vec.append(results["attn_vecs"][1][index])
            # label = results.labels[index]
            # prediction = results.predictions[index]

            figure = draw_attention_heatmap(phrases=phrases, attns=attn_vec)

        summary = figure_to_summary(name="Attention Heatmap", figure=figure)
        self._summary_writer.add_summary(summary, global_step)
