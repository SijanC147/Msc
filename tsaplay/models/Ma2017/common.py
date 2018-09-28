import tensorflow as tf
from tensorflow.python.keras.preprocessing import (  # pylint: disable=E0611
    sequence
)
from tsaplay.utils._data import (
    zip_list_join,
    zip_str_join,
    pad_for_dataset,
    package_feature_dict,
    prep_dataset_and_get_iterator,
    parse_tf_example,
)

params = {
    "batch_size": 25,
    "max_seq_length": 85,
    "n_out_classes": 3,
    "learning_rate": 0.1,
    "l2_weight": 1e-5,
    "momentum": 0.9,
    "keep_prob": 0.5,
    "hidden_units": 50,
    "initializer": tf.initializers.random_uniform(minval=-0.1, maxval=0.1),
}


def ian_pre_processing_fn(features, labels=None):
    processed_features = {
        "context": tf.sparse_concat(
            sp_inputs=[features["left"], features["right"]], axis=1
        ),
        "context_ids": tf.sparse_concat(
            sp_inputs=[features["left_ids"], features["right_ids"]], axis=1
        ),
        "target": features["target"],
        "target_ids": features["target_ids"],
    }
    if labels is None:
        return processed_features
    return processed_features, labels


def ian_input_fn(tfrecord, batch_size, _eval=False):
    shuffle_buffer = batch_size * 10
    dataset = tf.data.TFRecordDataset(tfrecord)
    dataset = dataset.map(parse_tf_example)
    dataset = dataset.map(ian_pre_processing_fn)

    if _eval:
        dataset = dataset.shuffle(buffer_size=shuffle_buffer)
    else:
        dataset = dataset.apply(
            tf.contrib.data.shuffle_and_repeat(buffer_size=shuffle_buffer)
        )

    dataset = dataset.batch(batch_size)

    return dataset.make_one_shot_iterator().get_next()


def ian_serving_fn(features):
    return ian_pre_processing_fn(features)
