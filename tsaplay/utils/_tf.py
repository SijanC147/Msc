import tensorflow as tf
from tensorflow.estimator import ModeKeys  # pylint: disable=E0401
import numpy as np
import matplotlib
import io
from tensorflow.contrib.lookup import (  # pylint: disable=E0611
    index_table_from_file
)
from tensorflow.contrib.layers import embed_sequence  # pylint: disable=E0611


def sparse_seq_lengths(sp_input, batched=True):
    if not (batched):
        batch_groups, _ = tf.unstack(sp_input.indicesl, axis=1)
    else:
        batch_groups, _, _ = tf.unstack(sp_input.indices, axis=1)
    _, _, counts = tf.unique_with_counts(batch_groups)

    return counts


def concat_seq_sparse(sp_inputs, axis, reverse=False):
    concatenated = tf.sparse_concat(sp_inputs=sp_inputs, axis=axis)
    indices = concatenated.indices

    batch_group, not_imp, _ = tf.unstack(indices, axis=1)
    _, _, counts = tf.unique_with_counts(batch_group)

    cumulutive = tf.cumsum(counts, exclusive=True, reverse=reverse)
    offsets = tf.gather(cumulutive, batch_group)

    num_values = tf.shape(batch_group)[0]
    if reverse:
        new_value_pos = tf.reverse(tf.range(num_values), axis=[0]) - offsets
    else:
        new_value_pos = tf.range(num_values) - offsets

    new_value_pos = tf.cast(new_value_pos, tf.int64)
    new_indices = tf.transpose(
        tf.stack([batch_group, not_imp, new_value_pos]), perm=[1, 0]
    )

    new_sparse = tf.SparseTensor(
        new_indices, concatenated.values, concatenated.dense_shape
    )

    if reverse:
        return tf.sparse_reorder(new_sparse)
    return new_sparse


def variable_len_batch_mean(input_tensor, seq_lengths, op_name):
    with tf.name_scope(name=op_name):
        input_sum = tf.reduce_sum(
            input_tensor=input_tensor, axis=1, keepdims=True
        )
        seq_lengths_t = tf.transpose([[seq_lengths]])
        seq_lengths_tiled = tf.tile(
            seq_lengths_t, multiples=[1, 1, tf.shape(input_sum)[2]]
        )
        seq_lengths_float = tf.to_float(seq_lengths_tiled)
        batched_means = tf.divide(input_sum, seq_lengths_float)

    return batched_means


def masked_softmax(logits, mask):
    """
    Masked softmax over dim 1, mask broadcasts over dim 2
    :param logits: (N, L, T)
    :param mask: (N, L)
    :return: probabilities (N, L, T)
    """
    v = tf.shape(logits)[2]
    indices = tf.cast(tf.where(tf.logical_not(mask)), tf.int32)
    inf = tf.constant(
        np.array([[tf.float32.max]], dtype=np.float32), dtype=tf.float32
    )
    infs = tf.tile(inf, [tf.shape(indices)[0], v])
    infmask = tf.scatter_nd(
        indices=indices, updates=infs, shape=tf.shape(logits)
    )
    masked_sm = tf.nn.softmax(logits - infmask, axis=1)

    return masked_sm


def gru_cell(hidden_units, inititalizer):
    return tf.nn.rnn_cell.GRUCell(
        num_units=hidden_units,
        kernel_initializer=inititalizer,
        bias_initializer=inititalizer,
    )


def dropout_gru_cell(hidden_units, initializer, keep_prob):
    return tf.contrib.rnn.DropoutWrapper(
        cell=gru_cell(hidden_units, initializer), output_keep_prob=keep_prob
    )


def lstm_cell(hidden_units, initializer):
    return tf.nn.rnn_cell.LSTMCell(
        num_units=hidden_units, initializer=initializer
    )


def dropout_lstm_cell(hidden_units, initializer, keep_prob):
    return tf.contrib.rnn.DropoutWrapper(
        cell=lstm_cell(hidden_units, initializer), output_keep_prob=keep_prob
    )


def l2_regularized_loss(
    labels,
    logits,
    l2_weight,
    variables=tf.trainable_variables(),
    loss_fn=tf.losses.sparse_softmax_cross_entropy,
):
    loss = loss_fn(labels=labels, logits=logits)
    l2_reg = tf.reduce_sum([tf.nn.l2_loss(v) for v in variables])
    loss = loss + l2_weight * l2_reg
    return loss


def attention_unit(
    h_states, hidden_units, seq_lengths, attn_focus, init, literal=None
):
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

    attn_summary_info = tf.tuple([literal, attn_vec])

    attn_vec = tf.expand_dims(attn_vec, axis=3)

    weighted_h_states = tf.einsum("Baij,Bajk->Baik", attn_vec, h_states)

    weighted_h_states_sum = tf.reduce_sum(
        input_tensor=weighted_h_states, axis=1
    )

    final_rep = tf.squeeze(input=weighted_h_states_sum, axis=1)

    return (
        final_rep,  # dim: [batch_size, hidden_units*2] (for BiLSTM)
        attn_summary_info,  # to optionally use for summary heatmaps
    )


def append_snapshot(container, new_snap, index):
    new_snap = tf.expand_dims(new_snap, axis=0)
    total_snaps = tf.shape(container)[0]
    batch_diff = tf.shape(container)[1] - tf.shape(new_snap)[1]
    new_snap = tf.pad(
        new_snap,
        paddings=[
            [index - 1, total_snaps - index],
            [0, batch_diff],
            [0, 0],
            [0, 0],
        ],
    )
    container = tf.add(container, new_snap)

    return container


def create_snapshots_container(shape_like, n_snaps):
    container = tf.zeros_like(shape_like, dtype=tf.float32)
    container = tf.expand_dims(container, axis=0)
    container = tf.expand_dims(container, axis=3)
    container = tf.tile(container, multiples=[n_snaps, 1, 1, 1])

    return container


def zip_attn_snapshots_with_literals(literals, snapshots, num_layers):
    max_len = tf.shape(snapshots)[2]
    snapshots = tf.transpose(snapshots, perm=[1, 0, 2, 3])
    snapshots = tf.reshape(snapshots, shape=[-1, max_len, 1])

    literals = tf.expand_dims(literals, axis=1)
    literals = tf.tile(literals, multiples=[1, num_layers])
    literals = tf.reshape(literals, shape=[-1])

    return literals, snapshots


def bulk_add_to_collection(collection, *variables):
    for variable in variables:
        tf.add_to_collection(collection, variable)


def generate_attn_heatmap_summary(*attn_infos):
    for attn_info in attn_infos:
        tf.add_to_collection("ATTENTION", attn_info)


def figure_to_summary(name, figure):
    # attach a new canvas if not exists
    if figure.canvas is None:
        matplotlib.backends.backend_agg.FigureCanvasAgg(figure)

    figure.canvas.draw()
    w, h = figure.canvas.get_width_height()

    # get PNG data from the figure
    png_buffer = io.BytesIO()
    figure.canvas.print_png(png_buffer)
    png_encoded = png_buffer.getvalue()
    png_buffer.close()

    summary_image = tf.Summary.Image(
        height=h,
        width=w,
        colorspace=4,  # RGB-A
        encoded_image_string=png_encoded,
    )
    summary = tf.Summary(
        value=[tf.Summary.Value(tag=name, image=summary_image)]
    )
    return summary


def image_to_summary(name, image):
    with io.BytesIO() as output:
        image.save(output, "PNG")
        png_encoded = output.getvalue()

    summary_image = tf.Summary.Image(
        height=image.size[1],
        width=image.size[0],
        colorspace=4,  # RGB-A
        encoded_image_string=png_encoded,
    )
    summary = tf.Summary(
        value=[tf.Summary.Value(tag=name, image=summary_image)]
    )
    return summary


def get_dense_tensor(sparse_tensor):
    dense_tensor = tf.sparse_to_dense(
        sparse_indices=sparse_tensor.indices,
        output_shape=sparse_tensor.dense_shape,
        sparse_values=sparse_tensor.values,
    )

    return dense_tensor


def setup_embedding_lookup_table(vocab_file_path):
    index_table = index_table_from_file(
        vocabulary_file=vocab_file_path, default_value=1
    )

    return index_table


def lookup_embedding_ids(index_table, string_tensors, back_pad=True):
    sparse_tokens_tensor = tf.string_split(string_tensors, delimiter="<SEP>")
    sparse_ids = index_table.lookup(sparse_tokens_tensor)

    dense_ids = get_dense_tensor(sparse_ids)

    if back_pad:
        batch_size = tf.shape(dense_ids)[0]
        padding = tf.zeros(shape=[batch_size, 1], dtype=tf.int64)
        dense_ids = tf.concat([dense_ids, padding], axis=1)

    return dense_ids


def setup_embedding_layer(
    vocab_size, dim_size, init, trainable=True, var_scope="embedding_layer"
):
    with tf.variable_scope(var_scope, reuse=tf.AUTO_REUSE):
        embeddings = tf.get_variable(
            "embeddings",
            shape=[vocab_size, dim_size],
            initializer=init,
            trainable=trainable,
        )

    return embeddings


def get_embedded_seq(
    ids, embedding_matrix, reuse=True, var_scope="embedding_layer"
):
    ids = tf.squeeze(ids, axis=1)
    embedded_seq = embed_sequence(
        ids=ids, initializer=embedding_matrix, scope=var_scope, reuse=True
    )

    return embedded_seq

