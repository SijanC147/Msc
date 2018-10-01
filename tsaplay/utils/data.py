import tensorflow as tf
from functools import wraps


def parse_tf_example(example):
    feature_spec = {
        "left": tf.VarLenFeature(dtype=tf.string),
        "target": tf.VarLenFeature(dtype=tf.string),
        "right": tf.VarLenFeature(dtype=tf.string),
        "left_ids": tf.VarLenFeature(dtype=tf.int64),
        "target_ids": tf.VarLenFeature(dtype=tf.int64),
        "right_ids": tf.VarLenFeature(dtype=tf.int64),
        "labels": tf.FixedLenFeature(dtype=tf.int64, shape=[]),
    }
    parsed_example = tf.parse_example([example], features=feature_spec)

    features = {
        "left": parsed_example["left"],
        "target": parsed_example["target"],
        "right": parsed_example["right"],
        "left_ids": parsed_example["left_ids"],
        "target_ids": parsed_example["target_ids"],
        "right_ids": parsed_example["right_ids"],
    }
    labels = tf.squeeze(parsed_example["labels"], axis=0)

    return (features, labels)


def prep_dataset(tfrecord, batch_size, processing_fn, mode):
    shuffle_buffer = batch_size * 10
    dataset = tf.data.TFRecordDataset(tfrecord)
    dataset = dataset.map(parse_tf_example)
    if processing_fn is not None:
        dataset = dataset.map(processing_fn)

    if mode == "EVAL":
        dataset = dataset.shuffle(buffer_size=shuffle_buffer)
    elif mode == "TRAIN":
        dataset = dataset.apply(
            tf.contrib.data.shuffle_and_repeat(buffer_size=shuffle_buffer)
        )

    dataset = dataset.batch(batch_size)

    return dataset


def make_input_fn(mode):
    def decorator(func):
        @wraps(func)
        def input_fn(*args, **kwargs):
            if mode == "TRAIN" or mode == "EVAL":
                try:
                    tfrecord = args[1]
                except IndexError:
                    tfrecord = kwargs.get("tfrecord")
                try:
                    batch_size = args[2]
                except IndexError:
                    batch_size = kwargs.get("batch_size")

                dataset = prep_dataset(
                    tfrecord=tfrecord,
                    batch_size=batch_size,
                    processing_fn=args[0].processing_fn,
                    mode=mode,
                )

                return dataset.make_one_shot_iterator().get_next()
            elif mode == "PREDICT":
                try:
                    features = args[1]
                except IndexError:
                    features = kwargs.get("features")
                return args[0].processing_fn(features)
            return input_fn

        return decorator
