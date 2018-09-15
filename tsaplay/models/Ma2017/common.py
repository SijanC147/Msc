import tensorflow as tf
from tensorflow.python.keras.preprocessing import (  # pylint: disable=E0611
    sequence
)
from tsaplay.utils._tf import masked_softmax

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


def ian_input_fn(
    features, labels, batch_size, max_seq_length, eval_input=False
):
    contexts = [
        l + r
        for l, r in zip(
            features["mappings"]["left"], features["mappings"]["right"]
        )
    ]
    contexts_len = [len(context) for context in contexts]
    contexts = sequence.pad_sequences(
        sequences=contexts,
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
        (contexts, contexts_len, targets, targets_len, labels)
    )
    dataset = dataset.map(
        lambda context, context_len, target, target_len, label: (
            {
                "context": {"x": context, "len": context_len},
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
