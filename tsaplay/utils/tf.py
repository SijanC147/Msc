import io
import numpy as np
import tensorflow as tf
from tensorflow.estimator import ModeKeys  # pylint: disable=E0401
from tensorflow.train import BytesList, Feature, Features, Example, Int64List
from tsaplay.constants import DELIMITER, MAX_EMBEDDING_SHARDS
from tsaplay.utils.io import cprnt, export_run_metadata
from tsaplay.utils.data import zero_norm_labels, split_list


def tf_class_distribution(labels):
    with tf.name_scope("debug_distribution_monitor"):
        ones = tf.ones_like(labels)
        zeros = tf.zeros_like(labels)

        negative = tf.reduce_sum(tf.where(tf.equal(labels, 0), ones, zeros))
        neutral = tf.reduce_sum(tf.where(tf.equal(labels, 1), ones, zeros))
        positive = tf.reduce_sum(tf.where(tf.equal(labels, 2), ones, zeros))

        distribution = tf.stack([negative, neutral, positive])
        totals = tf.reduce_sum(ones) * tf.ones_like(distribution)

        percentages = tf.divide(distribution, totals) * 100

    return tf.cast(percentages, tf.int32)


def scaffold_init_fn_on_spec(spec, new_fn):
    if spec.mode != ModeKeys.TRAIN:
        return spec
    scaffold = spec.scaffold or tf.train.Scaffold()
    prev_init = scaffold.init_fn

    def new_init_fn(scaffold, sess):
        if prev_init is not None:
            prev_init(sess)
        new_fn(sess)

    scaffold._init_fn = lambda sess: new_init_fn(scaffold, sess)
    return spec._replace(scaffold=scaffold)


def sparse_sequences_to_dense(sp_sequences):
    if sp_sequences.dtype == tf.string:
        default = b""
    else:
        default = 0
    dense = tf.sparse_tensor_to_dense(sp_sequences, default_value=default)
    needs_squeezing = tf.equal(tf.size(sp_sequences.dense_shape), 3)
    dense = tf.cond(
        needs_squeezing, lambda: tf.squeeze(dense, axis=1), lambda: dense
    )

    dense = tf.pad(dense, paddings=[[0, 0], [0, 1]], constant_values=default)

    return dense


def sparse_reverse(sp_input):
    reversed_indices = tf.reverse(sp_input.indices, axis=[0])
    reversed_sp_input = tf.SparseTensor(
        reversed_indices, sp_input.values, sp_input.dense_shape
    )
    return tf.sparse_reorder(reversed_sp_input)


def get_seq_lengths(batched_sequences):
    lengths = tf.reduce_sum(tf.sign(batched_sequences), axis=1)
    return tf.cast(lengths, tf.int32)


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
    seq_len = tf.shape(logits)[2]
    indices = tf.cast(tf.where(tf.logical_not(mask)), tf.int32)
    inf = tf.constant(
        np.array([[tf.float32.max]], dtype=np.float32), dtype=tf.float32
    )
    infs = tf.tile(inf, [tf.shape(indices)[0], seq_len])
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
    h_states, hidden_units, seq_lengths, attn_focus, init, sp_literal=None
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

    # literal_tensor = sparse_sequences_to_dense(sp_literal)
    # attn_summary_info = tf.tuple([literal_tensor, attn_vec])
    attn_summary_info = tf.tuple([sp_literal, attn_vec])

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


def zip_attn_snapshots_with_sp_literals(sp_literals, snapshots, num_layers):
    max_len = tf.shape(snapshots)[2]
    snapshots = tf.transpose(snapshots, perm=[1, 0, 2, 3])
    snapshots = tf.reshape(snapshots, shape=[-1, max_len, 1])

    literals = sparse_sequences_to_dense(sp_literals)
    literals = tf.expand_dims(literals, axis=1)
    literals = tf.tile(literals, multiples=[1, num_layers, 1])
    literals = tf.reshape(literals, shape=[-1, max_len])

    return literals, snapshots


def bulk_add_to_collection(collection, *variables):
    for variable in variables:
        tf.add_to_collection(collection, variable)


def generate_attn_heatmap_summary(*attn_infos):
    for attn_info in attn_infos:
        tf.add_to_collection("ATTENTION", attn_info)


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


def index_lookup_table(vocab_file, oov_buckets=0):
    return tf.contrib.lookup.index_table_from_file(
        vocabulary_file=vocab_file,
        key_column_index=0,
        value_column_index=1,
        num_oov_buckets=oov_buckets,
        delimiter="\t",
    )


def fetch_lookup_ops(tokens_lists, lookup_table):
    tokens_list = [
        DELIMITER.join(tkns_list).encode() for tkns_list in tokens_lists
    ]
    tokens_tensors = [
        tf.constant([tkns_list], dtype=tf.string) for tkns_list in tokens_list
    ]
    tokens_sp_tensors = [
        tf.string_split(tkn_ten, DELIMITER) for tkn_ten in tokens_tensors
    ]
    string_ops = [
        tf.sparse_tensor_to_dense(sp_tensor, default_value=b"")
        for sp_tensor in tokens_sp_tensors
    ]
    id_ops = [
        tf.sparse_tensor_to_dense(lookup_table.lookup(sp_tensor))
        for sp_tensor in tokens_sp_tensors
    ]
    return string_ops + id_ops


def run_lookups(fetch_dict, metadata_path=None, eager=False):
    if tf.executing_eagerly() and not eager:
        raise ValueError("Eager execution is not supported.")
    run_metadata = tf.RunMetadata()
    run_opts = (
        tf.RunOptions(
            trace_level=tf.RunOptions.FULL_TRACE  # pylint: disable=E1101
        )
        if metadata_path
        else None
    )

    with tf.Session() as sess:
        sess.run(tf.tables_initializer())
        values_dict = sess.run(
            fetch_dict, options=run_opts, run_metadata=run_metadata
        )

    if metadata_path:
        export_run_metadata(run_metadata, path=metadata_path)

    return {
        key: [value.tolist()[0] for value in values]
        for (key, values) in values_dict.items()
    }


def make_tf_examples(string_features, int_features, labels):
    int_features += [[label] for label in zero_norm_labels(labels)]
    string_features = [
        Feature(bytes_list=BytesList(value=val)) for val in string_features
    ]
    int_features = [
        Feature(int64_list=Int64List(value=val)) for val in int_features
    ]
    all_features = string_features + int_features
    return [
        Example(
            features=Features(
                feature={
                    "left": left,
                    "target": target,
                    "right": right,
                    "left_ids": left_ids,
                    "target_ids": target_ids,
                    "right_ids": right_ids,
                    "labels": label,
                }
            )
        )
        for (
            left,
            target,
            right,
            left_ids,
            target_ids,
            right_ids,
            label,
        ) in zip(*split_list(all_features, parts=7))
    ]


def partitioner_num_shards(vocab_size, max_shards=MAX_EMBEDDING_SHARDS):
    for i in range(max_shards, 0, -1):
        if vocab_size % i == 0:
            return i


def embedding_initializer(vectors, num_shards, structure=None):
    shape = vectors.shape
    partition_size = int(shape[0] / num_shards)

    def _init_part_var(shape=shape, dtype=tf.float32, partition_info=None):
        part_offset = partition_info.single_offset(shape)
        this_slice = part_offset + partition_size
        return vectors[part_offset:this_slice]

    def _init_var(shape=shape, dtype=tf.float32, partition_info=None):
        return vectors

    def _init_const():
        return vectors

    return {
        "partitioned": _init_part_var,
        "variable": _init_var,
        "constant": _init_const,
    }.get(structure, _init_part_var)
