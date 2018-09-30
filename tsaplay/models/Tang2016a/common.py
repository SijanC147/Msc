import tensorflow as tf
from tensorflow.python.keras.preprocessing import (  # pylint: disable=E0611
    sequence
)
from tsaplay.utils.data import (
    parse_tf_example,
    prep_dataset_and_get_iterator,
    zip_list_join,
    pad_for_dataset,
    package_feature_dict,
    make_labels_dataset_from_list,
    tf_encoded_tokenisation,
)
from tsaplay.utils.tf import sparse_reverse

params = {
    "batch_size": 25,
    "max_seq_length": 140,
    "n_out_classes": 3,
    "learning_rate": 0.01,
    "keep_prob": 0.8,
    "hidden_units": 200,
    "initializer": tf.initializers.random_uniform(minval=-0.03, maxval=0.03),
}


def lstm_pre_processing_fn(features, labels=None):
    processed_features = {
        "sentence_ids": tf.sparse_concat(
            sp_inputs=[
                features["left_ids"],
                features["target_ids"],
                features["right_ids"],
            ],
            axis=1,
        )
    }
    if labels is None:
        return processed_features
    return processed_features, labels


def lstm_input_fn(tfrecord, batch_size, _eval=False):
    shuffle_buffer = batch_size * 10
    dataset = tf.data.TFRecordDataset(tfrecord)
    dataset = dataset.map(parse_tf_example)
    dataset = dataset.map(lstm_pre_processing_fn)

    if _eval:
        dataset = dataset.shuffle(buffer_size=shuffle_buffer)
    else:
        dataset = dataset.apply(
            tf.contrib.data.shuffle_and_repeat(buffer_size=shuffle_buffer)
        )

    dataset = dataset.batch(batch_size)

    return dataset.make_one_shot_iterator().get_next()


def lstm_serving_fn(features):
    return lstm_pre_processing_fn(features)


def tdlstm_pre_processing_fn(features, labels=None):
    processed_features = {
        "left_ids": tf.sparse_concat(
            sp_inputs=[features["left_ids"], features["target_ids"]], axis=1
        ),
        "right_ids": sparse_reverse(
            tf.sparse_concat(
                sp_inputs=[features["right_ids"], features["target_ids"]],
                axis=1,
            )
        ),
    }
    if labels is None:
        return processed_features
    return processed_features, labels


def tdlstm_input_fn(tfrecord, batch_size, _eval=False):
    shuffle_buffer = batch_size * 10
    dataset = tf.data.TFRecordDataset(tfrecord)
    dataset = dataset.map(parse_tf_example)
    dataset = dataset.map(tdlstm_pre_processing_fn)

    if _eval:
        dataset = dataset.shuffle(buffer_size=shuffle_buffer)
    else:
        dataset = dataset.apply(
            tf.contrib.data.shuffle_and_repeat(buffer_size=shuffle_buffer)
        )

    dataset = dataset.batch(batch_size)

    return dataset.make_one_shot_iterator().get_next()


def tdlstm_serving_fn(features):
    return tdlstm_pre_processing_fn(features)


def tclstm_pre_processing_fn(features, labels=None):
    processed_features = {
        "left_ids": tf.sparse_concat(
            sp_inputs=[features["left_ids"], features["target_ids"]], axis=1
        ),
        "right_ids": sparse_reverse(
            tf.sparse_concat(
                sp_inputs=[features["right_ids"], features["target_ids"]],
                axis=1,
            )
        ),
        "target_ids": features["target_ids"],
    }
    if labels is None:
        return processed_features
    return processed_features, labels


def tclstm_input_fn(tfrecord, batch_size, _eval=False):
    shuffle_buffer = batch_size * 10
    dataset = tf.data.TFRecordDataset(tfrecord)
    dataset = dataset.map(parse_tf_example)
    dataset = dataset.map(tclstm_pre_processing_fn)

    if _eval:
        dataset = dataset.shuffle(buffer_size=shuffle_buffer)
    else:
        dataset = dataset.apply(
            tf.contrib.data.shuffle_and_repeat(buffer_size=shuffle_buffer)
        )

    dataset = dataset.batch(batch_size)

    return dataset.make_one_shot_iterator().get_next()


def tclstm_serving_fn(features):
    return tclstm_pre_processing_fn(features)
