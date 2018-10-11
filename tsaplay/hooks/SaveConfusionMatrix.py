import textwrap
import re
import itertools

import matplotlib
import numpy as np
import tensorflow as tf
from tensorflow.train import SessionRunHook
from tsaplay.utils.tf import image_to_summary
from tsaplay.utils.io import get_image_from_plt, temp_pngs, cprnt

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt  # nopep8


class SaveConfusionMatrix(SessionRunHook):
    """
    Saves a confusion matrix as a Summary so that it can be shown in 
    tensorboard
    """

    def __init__(
        self,
        class_labels,
        confusion_matrix_tensor_name,
        summary_writer,
        comet=None,
    ):
        """Initializes a `SaveConfusionMatrixHook`.

        :param labels: Iterable of String containing the labels to print for
                        each row/column in the confusion matrix.
        :param confusion_matrix_tensor_name: The name of the tensor containing
                                            the confusionmatrix
        :param summary_writer: The summary writer that will save the summary
        """
        self.confusion_matrix_tensor_name = confusion_matrix_tensor_name
        self.class_labels = class_labels
        self._summary_writer = summary_writer
        self._comet = comet

    def end(self, session):
        confusion_matrix = (
            tf.get_default_graph()
            .get_tensor_by_name(self.confusion_matrix_tensor_name + ":0")
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

    def _plot_confusion_matrix(self, cm):
        """
        :param cm: A confusion matrix: A square ```numpy array``` of the same
                                        size as self.labels
        :return:  A ``matplotlib.figure.Figure`` object with a numerical and
                                        graphical representation of the cm
                                        array
        """
        num_classes = len(self.class_labels)

        fig = plt.figure(
            figsize=(num_classes, num_classes), facecolor="w", edgecolor="k"
        )
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(cm, cmap="Oranges")

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
                int(cm[i, j]) if cm[i, j] != 0 else ".",
                horizontalalignment="center",
                verticalalignment="center",
                color="black",
            )
        fig.set_tight_layout(True)

        image = get_image_from_plt(plt)
        return image
