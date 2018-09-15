import tensorflow as tf
from tensorflow.python.keras.preprocessing import (  # pylint: disable=E0611
    sequence
)

params = {
    "batch_size": 1,
    "max_seq_length": 25,
    "n_out_classes": 3,
    "learning_rate": 0.01,
    "keep_prob": 0.8,
    "hidden_units": 5,
    "initializer": tf.initializers.random_uniform(minval=-0.03, maxval=0.03),
}


def lstm_input_fn(
    features, labels, batch_size, max_seq_length, eval_input=False
):
    sentences = [
        l + t + r
        for l, t, r in zip(
            features["mappings"]["left"],
            features["mappings"]["target"],
            features["mappings"]["right"],
        )
    ]
    sens_lens = [
        len(l + t + r)
        for l, t, r in zip(
            features["mappings"]["left"],
            features["mappings"]["target"],
            features["mappings"]["right"],
        )
    ]
    labels = [label + 1 for label in labels]

    sentences = sequence.pad_sequences(
        sequences=sentences,
        maxlen=max_seq_length,
        truncating="post",
        padding="post",
        value=0,
    )

    dataset = tf.data.Dataset.from_tensor_slices(
        (sentences, sens_lens, labels)
    )
    dataset = dataset.map(
        lambda sentence, length, label: ({"x": sentence, "len": length}, label)
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


def tdlstm_input_fn(
    features, labels, batch_size, max_seq_length, eval_input=False
):
    left_contexts = [
        l + t
        for l, t in zip(
            features["mappings"]["left"], features["mappings"]["target"]
        )
    ]
    left_contexts_len = [len(left_context) for left_context in left_contexts]
    left_contexts = sequence.pad_sequences(
        sequences=left_contexts,
        maxlen=max_seq_length,
        truncating="post",
        padding="post",
        value=0,
    )

    right_contexts = [
        list(reversed(t + r))
        for t, r in zip(
            features["mappings"]["target"], features["mappings"]["right"]
        )
    ]
    right_contexts_len = [
        len(right_context) for right_context in right_contexts
    ]
    right_contexts = sequence.pad_sequences(
        sequences=right_contexts,
        maxlen=max_seq_length,
        truncating="post",
        padding="post",
        value=0,
    )

    labels = [label + 1 for label in labels]

    dataset = tf.data.Dataset.from_tensor_slices(
        (
            left_contexts,
            left_contexts_len,
            right_contexts,
            right_contexts_len,
            labels,
        )
    )
    dataset = dataset.map(
        lambda left, left_len, right, right_len, label: (
            {
                "left": {"x": left, "len": left_len},
                "right": {"x": right, "len": right_len},
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


def tclstm_input_fn(
    features, labels, batch_size, max_seq_length, eval_input=False
):
    left_contexts = [
        l + t
        for l, t in zip(
            features["mappings"]["left"], features["mappings"]["target"]
        )
    ]
    left_contexts_len = [len(left_context) for left_context in left_contexts]
    left_contexts = sequence.pad_sequences(
        sequences=left_contexts,
        maxlen=max_seq_length,
        truncating="post",
        padding="post",
        value=0,
    )

    right_contexts = [
        list(reversed(t + r))
        for t, r in zip(
            features["mappings"]["target"], features["mappings"]["right"]
        )
    ]
    right_contexts_len = [
        len(right_context) for right_context in right_contexts
    ]
    right_contexts = sequence.pad_sequences(
        sequences=right_contexts,
        maxlen=max_seq_length,
        truncating="post",
        padding="post",
        value=0,
    )

    targets = [t for t in features["mappings"]["target"]]
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
            left_contexts,
            left_contexts_len,
            right_contexts,
            right_contexts_len,
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
