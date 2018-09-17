import tensorflow as tf
from tensorflow.train import SessionRunHook, SessionRunArgs, get_global_step


class LoggingHook(SessionRunHook):
    def begin(self):
        self._global_step = get_global_step()

    def before_run(self, run_context):
        graph = tf.get_default_graph()
        return SessionRunArgs(
            fetches={
                "global_step": tf.get_collection(tf.GraphKeys.GLOBAL_STEP),
                "losses": tf.get_collection(tf.GraphKeys.LOSSES),
                "metrics": tf.get_collection(tf.GraphKeys.METRIC_VARIABLES),
                "queue": tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS),
                # "model": tf.get_collection(tf.GraphKeys.MODEL_VARIABLES),
                "res": tf.get_collection(tf.GraphKeys.RESOURCES),
                "act": tf.get_collection(tf.GraphKeys.ACTIVATIONS),
                "path": tf.get_collection(tf.GraphKeys.ASSET_FILEPATHS),
                "concat": tf.get_collection(
                    tf.GraphKeys.CONCATENATED_VARIABLES
                ),
                "ready": tf.get_collection(
                    tf.GraphKeys.READY_FOR_LOCAL_INIT_OP
                ),
                "saveable": tf.get_collection(tf.GraphKeys.SAVEABLE_OBJECTS),
                "vars": tf.get_collection(
                    tf.GraphKeys.VARIABLES, scope="logits"
                ),
                "labels": tf.get_collection(
                    tf.GraphKeys.GLOBAL_VARIABLES, scope="IteratorGetNext"
                ),
                "trres": tf.get_collection(
                    tf.GraphKeys.TRAINABLE_RESOURCE_VARIABLES
                ),
                # "savers": tf.get_collection(tf.GraphKeys.SAVERS)
                # "cond_ctxt": tf.get_collection(tf.GraphKeys.COND_CONTEXT)
                # "all": tf.get_collegtion(tf.GraphKeys.GLOBAL_VARIABLES, ),
            }
        )

    def after_run(self, run_context, run_values):
        print(run_values.results)

