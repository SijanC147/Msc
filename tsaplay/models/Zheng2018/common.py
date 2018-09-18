import tensorflow as tf
from tensorflow.python.keras.preprocessing import (  # pylint: disable=E0611
    sequence
)

params = {
    "batch_size": 25,
    "max_seq_length": 85,
    "n_out_classes": 3,
    "learning_rate": 0.1,
    "l2_weight": 1e-5,
    "momentum": 0.9,
    "keep_prob": 0.5,
    "hidden_units": 100,
    "initializer": tf.initializers.random_uniform(minval=-0.1, maxval=0.1),
}


def lcr_rot_input_fn(
    features, labels, batch_size, max_seq_length, eval_input=False
):

    left_ctxts_lit = features["left"]
    left_ctxts_map = features["mappings"]["left"]
    left_ctxts_len = [len(l_ctxt) for l_ctxt in left_ctxts_map]
    left_ctxts_map = sequence.pad_sequences(
        sequences=left_ctxts_map,
        maxlen=max_seq_length,
        truncating="post",
        padding="post",
        value=0,
    )

    right_ctxts_lit = features["right"]
    right_ctxts_map = features["mappings"]["right"]
    right_ctxts_len = [len(r_ctxt) for r_ctxt in right_ctxts_map]
    right_ctxts_map = sequence.pad_sequences(
        sequences=right_ctxts_map,
        maxlen=max_seq_length,
        truncating="post",
        padding="post",
        value=0,
    )

    targets_lit = features["target"]
    targets_map = features["mappings"]["target"]
    targets_len = [len(t) for t in targets_map]
    targets_map = sequence.pad_sequences(
        sequences=targets_map,
        maxlen=max(targets_len),
        truncating="post",
        padding="post",
        value=0,
    )

    labels = [label + 1 for label in labels]

    dataset = tf.data.Dataset.from_tensor_slices(
        (
            left_ctxts_lit,
            left_ctxts_map,
            left_ctxts_len,
            right_ctxts_lit,
            right_ctxts_map,
            right_ctxts_len,
            targets_lit,
            targets_map,
            targets_len,
            labels,
        )
    )
    dataset = dataset.map(
        lambda l_lit, l_map, l_len, r_lit, r_map, r_len, t_lit, t_map, t_len, label: (  # nopep8
            {
                "left": {"x": l_map, "len": l_len, "lit": l_lit},
                "right": {"x": r_map, "len": r_len, "lit": r_lit},
                "target": {"x": t_map, "len": t_len, "lit": t_lit},
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

