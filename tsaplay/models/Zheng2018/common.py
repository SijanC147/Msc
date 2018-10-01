import tensorflow as tf
from tensorflow.python.keras.preprocessing import (  # pylint: disable=E0611
    sequence
)
from tsaplay.utils.data import parse_tf_example

params = {
    "batch_size": 50,
    "n_out_classes": 3,
    "learning_rate": 0.1,
    "l2_weight": 1e-5,
    "momentum": 0.9,
    "keep_prob": 0.5,
    "hidden_units": 200,
    "initializer": tf.initializers.random_uniform(minval=-0.1, maxval=0.1),
    "n_attn_heatmaps": 5,
}


def lcr_rot_input_fn(tfrecord, batch_size, eval_input=False):
    shuffle_buffer = batch_size * 100
    dataset = tf.data.TFRecordDataset(tfrecord)
    dataset = dataset.map(parse_tf_example)

    if eval_input:
        dataset = dataset.shuffle(buffer_size=shuffle_buffer)
    else:
        dataset = dataset.apply(
            tf.contrib.data.shuffle_and_repeat(buffer_size=shuffle_buffer)
        )

    dataset = dataset.batch(batch_size)

    return dataset.make_one_shot_iterator().get_next()


def lcr_rot_serving_fn(features):
    return features
