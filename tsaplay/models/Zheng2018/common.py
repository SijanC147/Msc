import tensorflow as tf
from tensorflow.python.keras.preprocessing import (  # pylint: disable=E0611
    sequence
)

params = {
    "batch_size": 3,
    "max_seq_length": 85,
    "n_out_classes": 3,
    "learning_rate": 0.1,
    "keep_prob": 0.5,
    "hidden_units": 2,
    "lstm_initializer": tf.initializers.random_uniform(
        minval=-0.1, maxval=0.1
    ),
}


def lstm_cell(params):
    return tf.nn.rnn_cell.LSTMCell(
        num_units=params["hidden_units"],
        initializer=params.get("lstm_initializer"),
    )


def dropout_lstm_cell(params):
    return tf.contrib.rnn.DropoutWrapper(
        cell=lstm_cell(params), output_keep_prob=params["keep_prob"]
    )


def lcr_rot_input_fn(
    features, labels, batch_size, max_seq_length, eval_input=False
):
    left_ctxts = features["mappings"]["left"]
    left_ctxts_len = [len(l_ctxt) for l_ctxt in left_ctxts]
    left_ctxts = sequence.pad_sequences(
        sequences=left_ctxts,
        maxlen=max_seq_length,
        truncating="post",
        padding="post",
        value=0,
    )

    right_ctxts = features["mappings"]["right"]
    right_ctxts_len = [len(r_ctxt) for r_ctxt in right_ctxts]
    right_ctxts = sequence.pad_sequences(
        sequences=right_ctxts,
        maxlen=max_seq_length,
        truncating="post",
        padding="post",
        value=0,
    )

    targets = features["mappings"]["target"]
    targets_len = [len(t) for t in targets]
    targets = sequence.pad_sequences(
        sequences=targets,
        maxlen=max(targets_len),
        truncating="post",
        padding="post",
        value=0,
    )

    labels = [label + 1 for label in labels]

    dataset = tf.data.Dataset.from_tensor_slices(
        (
            left_ctxts,
            left_ctxts_len,
            right_ctxts,
            right_ctxts_len,
            targets,
            targets_len,
            labels,
        )
    )
    dataset = dataset.map(
        lambda left, left_len, right, right_len, target, target_len, label: (
            {
                "left": {"x": left, "len": left_len},
                "right": {"x": right, "len": right_len},
                "target": {"x": target, "len": target_len},
            },
            label,
        )
    )

    if eval_input:
        dataset = dataset.shuffle(buffer_size=len(labels))
    else:
        dataset = dataset.apply(
            tf.contrib.data.shuffle_and_repeat(buffer_size=len(labels))
        )

    dataset = dataset.batch(batch_size=batch_size)

    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()
