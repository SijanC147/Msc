import tensorflow as tf
from tensorflow.estimator import ModeKeys  # pylint: disable=E0401
from tensorflow.train import SessionRunHook
from tsaplay.utils.io import cprnt


class CometContextSwitch(SessionRunHook):
    def __init__(self, comet, mode):
        self.comet = comet
        self.mode = mode

    def before_run(self, run_context):
        if self.comet.context != self.mode:
            cprnt(
                "Switching comet context from {0} to {1}".format(
                    self.comet.context, self.mode
                )
            )
            self.comet.context = self.mode

    # def end(self, session):
    #     if self.comet.context == ModeKeys.EVAL:
    #         self.comet.context = ModeKeys.TRAIN
