# pylint: disable=no-name-in-module
import io
from functools import wraps
from itertools import tee, chain
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from tensorflow.estimator import ModeKeys  # noqa
from tensorflow.train import BytesList, Feature, Features, Example, Int64List # noqa
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import ops
from tensorflow.contrib.metrics import confusion_matrix as cm
from tsaplay.constants import TF_DELIMITER, MAX_EMBEDDING_SHARDS
from tsaplay.utils.io import export_run_metadata, cprnt 
from tsaplay.utils.data import zero_norm_labels, split_list
from tsaplay.utils.debug import timeit


def embed_sequences(model_fn):
    @wraps(model_fn)
    def wrapper(self, features, labels, mode, params):
        embedded_sequences = {}
        embedding_init = self.aux_config.get("embedding_init", "partitioned")
        embedding_init_fn = params["_embedding_init"]
        num_shards = params["_embedding_num_shards"]
        embedding_partitioner = (
            tf.fixed_size_partitioner(num_shards)
            if embedding_init == "partitioned"
            else None
        )
        embedding_initializer = (
            params["_embedding_init"]
            if embedding_init in ["variable", "partitioned"]
            else None
        )
        vocab_size = params["_vocab_size"]
        dim_size = params["_embedding_dim"]
        trainable = params.get("train_embeddings", True)
        with tf.variable_scope("embedding_layer", reuse=tf.AUTO_REUSE):
            embeddings = tf.get_variable(
                "embeddings",
                shape=[vocab_size, dim_size],
                initializer=embedding_initializer,
                partitioner=embedding_partitioner,
                trainable=trainable,
                dtype=tf.float32,
            )
        for key, value in features.items():
            if "_ids" in key:
                component = key.replace("_ids", "")
                embdd_key = component + "_emb"
                embedded_sequence = tf.nn.embedding_lookup(
                    params=embeddings, ids=value, partition_strategy="div"
                )
                embedded_sequences[embdd_key] = embedded_sequence
        features.update(embedded_sequences)
        spec = model_fn(self, features, labels, mode, params)
        if embedding_init == "constant":

            def init_embeddings(sess):
                sess.run(
                    embeddings.initializer,
                    {embeddings.initial_value: embedding_init_fn()},
                )

            spec = scaffold_init_fn_on_spec(spec, init_embeddings)

        return spec

    return wrapper


def sharded_saver(model_fn):
    @wraps(model_fn)
    def wrapper(self, features, labels, mode, params):
        spec = model_fn(self, features, labels, mode, params)
        scaffold = spec.scaffold or tf.train.Scaffold()
        scaffold._saver = tf.train.Saver(  # pylint: disable=W0212
            sharded=True,
            max_to_keep=self.run_config.keep_checkpoint_max,
            keep_checkpoint_every_n_hours=self.run_config.keep_checkpoint_every_n_hours,
        )
        spec = spec._replace(scaffold=scaffold)

        return spec

    return wrapper


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


def make_dense_features(features):
    dense_features = {}
    for key in features:
        if "_ids" in key:
            name, _, _ = key.partition("_")
            if features.get(name):
                dense_features.update(
                    {name: sparse_sequences_to_dense(features[name])}
                )
            name_ids = sparse_sequences_to_dense(features[key])
            name_lens = get_seq_lengths(name_ids)
            dense_features.update(
                {name + "_ids": name_ids, name + "_len": name_lens}
            )
    features.update(dense_features)
    return features


def make_input_fn(mode):
    def decorator(func):
        @wraps(func)
        def input_fn(*args, **kwargs):
            if mode in ["TRAIN", "EVAL"]:
                try:
                    tfrecords = args[1]
                except IndexError:
                    tfrecords = kwargs.get("tfrecords")
                try:
                    params = args[2]
                except IndexError:
                    params = kwargs.get("params")

                def process_dataset(features, labels):
                    return (args[0].processing_fn(features), labels)

                return prep_dataset(
                    tfrecords=tfrecords,
                    params=params,
                    processing_fn=process_dataset,
                    mode=mode,
                )

            raise ValueError("Invalid mode: {0}".format(mode))

        return input_fn

    return decorator


def prep_dataset(tfrecords, params, processing_fn, mode):
    shuffle_buffer = params.get("shuffle_buffer", 100000)
    parallel_calls = params.get("parallel_calls", 4)
    parallel_batches = params.get("parallel_batches", parallel_calls)
    prefetch_buffer = params.get("prefetch_buffer", 100)
    dataset = tf.data.Dataset.list_files(file_pattern=tfrecords)
    dataset = dataset.apply(
        tf.data.experimental.parallel_interleave(
            tf.data.TFRecordDataset,
            cycle_length=3,
            buffer_output_elements=prefetch_buffer,
            prefetch_input_elements=parallel_calls,
        )
    )
    dataset = dataset.shuffle(
        buffer_size=shuffle_buffer, reshuffle_each_iteration=True
    )
    dataset = dataset.apply(
        tf.data.experimental.map_and_batch(
            lambda example: processing_fn(*parse_tf_example(example)),
            params["batch-size"],
            num_parallel_batches=parallel_batches,
        )
    )
    dataset = dataset.map(
        lambda features, labels: (make_dense_features(features), labels),
        num_parallel_calls=parallel_calls,
    )
    if mode == "TRAIN":
        #! epochs==0 => repeat indefinitely
        dataset = dataset.repeat(count=(params.get("epochs") or None))

    dataset = dataset.cache()

    return dataset


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
    dense = tf.sparse.to_dense(sp_sequences, default_value=default)
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


def gru_cell(**params):
    hidden_units = params.get("gru_hidden_units", params.get("hidden_units"))
    initializer = params.get("gru_initializer", params.get("initializer"))
    bias_initializer = params.get(
        "gru_bias_initializer", params.get("bias_initializer")
    )
    keep_prob = params.get("gru_keep_prob", params.get("keep_prob"))
    gru = tf.nn.rnn_cell.GRUCell(
        num_units=hidden_units,
        kernel_initializer=initializer,
        bias_initializer=(bias_initializer or initializer),
    )
    return (
        gru
        if not keep_prob
        else tf.contrib.rnn.DropoutWrapper(
            cell=gru, output_keep_prob=keep_prob
        )
    )


def lstm_cell(**params):
    hidden_units = params.get("lstm_hidden_units", params.get("hidden_units"))
    initializer = params.get("lstm_initializer", params.get("initializer"))
    initial_bias = params.get("lstm_initial_bias", 1)
    keep_prob = params.get("lstm_keep_prob", params.get("keep_prob"))
    lstm = tf.nn.rnn_cell.LSTMCell(
        num_units=hidden_units,
        initializer=initializer,
        forget_bias=initial_bias,
    )
    return (
        lstm
        if not keep_prob
        else tf.contrib.rnn.DropoutWrapper(
            cell=lstm, output_keep_prob=keep_prob
        )
    )


def l2_regularized_loss(
    labels,
    logits,
    l2_weight,
    variables=tf.trainable_variables(),
    loss_fn=tf.losses.sparse_softmax_cross_entropy,
):
    with tf.name_scope("l2_loss"):
        loss = loss_fn(labels=labels, logits=logits)
        l2_reg = tf.reduce_sum(
            [tf.nn.l2_loss(v) for v in variables], name="l2_reg"
        )
        loss = loss + l2_weight * l2_reg
    return loss


def attention_unit(
    h_states,
    hidden_units,
    seq_lengths,
    attn_focus,
    init,
    bias_init=None,
    sp_literal=None,
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
        name="bias",
        shape=[1],
        dtype=tf.float32,
        initializer=(bias_init or init),
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


def zip_attn_snapshots_with_literals(literals, snapshots, num_layers):
    max_len = tf.shape(snapshots)[2]
    snapshots = tf.transpose(snapshots, perm=[1, 0, 2, 3])
    snapshots = tf.reshape(snapshots, shape=[-1, max_len, 1])

    # literals = sparse_sequences_to_dense(sp_literals)
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


def ids_lookup_table(vocab_file_path, oov_buckets=1):
    return tf.contrib.lookup.index_table_from_file(
        vocabulary_file=vocab_file_path,
        key_column_index=0,
        value_column_index=1,
        num_oov_buckets=oov_buckets,
        delimiter="\t",
    )


def fetch_lookup_ops(lookup_table, **tokens_lists):
    list_lengths = [len(tkn_list) for tkn_list in tokens_lists.values()]
    tkns_lsts = sum([tkn_list for tkn_list in tokens_lists.values()], [])

    tokens_list = (
        TF_DELIMITER.join(tkns_list).encode("utf-8") for tkns_list in tkns_lsts
    )
    tokens_tensors = (
        tf.constant([tkns_list], dtype=tf.string) for tkns_list in tokens_list
    )
    tokens_sp_tensors = (
        tf.string_split(tkn_ten, TF_DELIMITER) for tkn_ten in tokens_tensors
    )
    string_sp_tensors, id_sp_tensors = tee(tokens_sp_tensors)
    string_ops = (
        tf.sparse.to_dense(sp_tensor, default_value=b"")
        for sp_tensor in string_sp_tensors
    )
    id_ops = (
        tf.sparse.to_dense(lookup_table.lookup(sp_tensor))
        for sp_tensor in id_sp_tensors
    )
    op_generator = chain(string_ops, id_ops)

    total = len(tkns_lsts) * 2
    op_generator = tqdm(op_generator, total=total, desc="Building Lookup Ops")
    ops = [op for op in op_generator]

    string_ops, id_ops = split_list(ops, parts=2)

    return {
        key: str_ops + id_ops
        for (key, str_ops, id_ops) in zip(
            [*tokens_lists],
            split_list(string_ops, counts=list_lengths),
            split_list(id_ops, counts=list_lengths),
        )
    }


@timeit("Executing token ID lookups", "Token IDs generated")
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
    return 1


def embedding_initializer_fn(vectors, num_shards, structure=None):
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


def metric_variable(shape, dtype, validate_shape=True, name=None):
    """Create variable in `GraphKeys.(LOCAL|METRIC_VARIABLES`) collections.
    from https://github.com/tensorflow/tensorflow/blob/r1.8/tensorflow/python/ops/metrics_impl.py
    """
    return variable_scope.variable(
        lambda: array_ops.zeros(shape, dtype),
        trainable=False,
        collections=[
            ops.GraphKeys.LOCAL_VARIABLES,
            ops.GraphKeys.METRIC_VARIABLES,
        ],
        validate_shape=validate_shape,
        name=name,
    )


def streaming_conf_matrix(labels, predictions, num_classes):
    conf_mat = metric_variable(
        shape=[num_classes, num_classes],
        dtype=tf.int64,
        validate_shape=False,
        name="total_confusion_matrix",
    )
    up_conf_mat = tf.assign_add(
        conf_mat,
        cm(labels, predictions, dtype=tf.int64, num_classes=num_classes),
    )

    return conf_mat, up_conf_mat


# pylint: disable=too-many-locals
def streaming_f1_scores(labels, predictions, num_classes):
    y_true = tf.cast(
        tf.one_hot(indices=labels, depth=num_classes), dtype=tf.int64
    )
    y_pred = tf.cast(
        tf.one_hot(indices=predictions, depth=num_classes), dtype=tf.int64
    )

    weights = metric_variable(
        shape=[num_classes],
        dtype=tf.int64,
        validate_shape=False,
        name="weights",
    )
    tp_mac = metric_variable(
        shape=[num_classes],
        dtype=tf.int64,
        validate_shape=False,
        name="tp_mac",
    )
    fp_mac = metric_variable(
        shape=[num_classes],
        dtype=tf.int64,
        validate_shape=False,
        name="fp_mac",
    )
    fn_mac = metric_variable(
        shape=[num_classes],
        dtype=tf.int64,
        validate_shape=False,
        name="fn_mac",
    )
    tp_mic = metric_variable(
        shape=[], dtype=tf.int64, validate_shape=False, name="tp_mic"
    )
    fp_mic = metric_variable(
        shape=[], dtype=tf.int64, validate_shape=False, name="fp_mic"
    )
    fn_mic = metric_variable(
        shape=[], dtype=tf.int64, validate_shape=False, name="fn_mic"
    )

    up_tp_mac = tf.assign_add(
        tp_mac, tf.count_nonzero(y_pred * y_true, axis=0)
    )
    up_fp_mac = tf.assign_add(
        fp_mac, tf.count_nonzero(y_pred * (y_true - 1), axis=0)
    )
    up_fn_mac = tf.assign_add(
        fn_mac, tf.count_nonzero((y_pred - 1) * y_true, axis=0)
    )
    up_tp_mic = tf.assign_add(
        tp_mic, tf.count_nonzero(y_pred * y_true, axis=None)
    )
    up_fp_mic = tf.assign_add(
        fp_mic, tf.count_nonzero(y_pred * (y_true - 1), axis=None)
    )
    up_fn_mic = tf.assign_add(
        fn_mic, tf.count_nonzero((y_pred - 1) * y_true, axis=None)
    )
    up_weights = tf.assign_add(weights, tf.reduce_sum(y_true, axis=0))

    updates = tf.group(
        up_tp_mac,
        up_fp_mac,
        up_fn_mac,
        up_tp_mic,
        up_fp_mic,
        up_fn_mic,
        up_weights,
    )

    weights = weights / tf.reduce_sum(weights)
    prec_mic = tp_mic / (tp_mic + fp_mic)
    prec_mic = tf.where(tf.is_nan(prec_mic), tf.zeros_like(prec_mic), prec_mic)
    rec_mic = tp_mic / (tp_mic + fn_mic)
    rec_mic = tf.where(tf.is_nan(rec_mic), tf.zeros_like(rec_mic), rec_mic)
    f1_mic = 2 * prec_mic * rec_mic / (prec_mic + rec_mic)
    f1_mic = tf.where(tf.is_nan(f1_mic), tf.zeros_like(f1_mic), f1_mic)
    f1_mic = tf.reduce_mean(f1_mic)
    prec_mac = tp_mac / (tp_mac + fp_mac)
    prec_mac = tf.where(tf.is_nan(prec_mac), tf.zeros_like(prec_mac), prec_mac)
    rec_mac = tp_mac / (tp_mac + fn_mac)
    rec_mac = tf.where(tf.is_nan(rec_mac), tf.zeros_like(rec_mac), rec_mac)
    f1_mac = 2 * prec_mac * rec_mac / (prec_mac + rec_mac)
    f1_mac = tf.where(tf.is_nan(f1_mac), tf.zeros_like(f1_mac), f1_mac)
    f1_wei = tf.reduce_sum(f1_mac * weights)
    f1_mac = tf.reduce_mean(f1_mac)

    return {
        "micro-f1": (f1_mic, updates),
        "macro-f1": (f1_mac, updates),
        "weighted-f1": (f1_wei, updates),
    }
