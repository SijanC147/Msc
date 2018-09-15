import tensorflow as tf
from tensorflow.python.keras.preprocessing import (  # pylint: disable=E0611
    sequence
)
from tsaplay.utils.common import masked_softmax

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


def lstm_cell(params):
    return tf.nn.rnn_cell.LSTMCell(
        num_units=params["hidden_units"], initializer=params.get("initializer")
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


def attention_unit(h_states, hidden_units, seq_lengths, attn_focus, init):
    batch_size = tf.shape(h_states)[0]
    max_seq_len = tf.shape(h_states)[1]
    weights = tf.get_variable(
        name="weights",
        shape=[hidden_units, hidden_units],
        dtype=tf.float32,
        initializer=init,
    )
    bias = tf.get_variable(
        name="bias", shape=[1], dtype=tf.float32, initializer=init
    )

    weights = tf.expand_dims(input=weights, axis=0)
    weights = tf.tile(
        input=weights, multiples=[batch_size * max_seq_len, 1, 1]
    )
    weights = tf.reshape(
        tensor=weights, shape=[-1, max_seq_len, hidden_units, hidden_units]
    )

    h_states = tf.expand_dims(input=h_states, axis=2)

    attn_focus = tf.tile(input=attn_focus, multiples=[1, max_seq_len, 1])
    attn_focus = tf.expand_dims(input=attn_focus, axis=3)

    bias_mask = tf.sequence_mask(
        lengths=seq_lengths, maxlen=max_seq_len, dtype=tf.float32
    )

    bias = bias_mask * bias
    bias = tf.reshape(tensor=bias, shape=[batch_size, -1, 1, 1])

    f_score = tf.nn.tanh(
        tf.einsum("Baij,Bajk,Bakn->Bain", h_states, weights, attn_focus) + bias
    )
    f_score = tf.squeeze(input=f_score, axis=3)

    mask = tf.sequence_mask(lengths=seq_lengths, maxlen=max_seq_len)

    attn_vec = masked_softmax(logits=f_score, mask=mask)

    attn_vec = tf.expand_dims(attn_vec, axis=3)

    weighted_h_states = tf.einsum("Baij,Bajk->Baik", attn_vec, h_states)

    weighted_h_states_sum = tf.reduce_sum(
        input_tensor=weighted_h_states, axis=1
    )

    final_rep = tf.squeeze(input=weighted_h_states_sum, axis=1)

    return final_rep  # dim: [batch_size, hidden_units*2] (for BiLSTM)
