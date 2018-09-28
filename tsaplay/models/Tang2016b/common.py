import tensorflow as tf
from math import ceil
from tensorflow.python.keras.preprocessing import (  # pylint: disable=E0611
    sequence
)
from tsaplay.utils._data import (
    parse_tf_example,
    zip_list_join,
    zip_str_join,
    pad_for_dataset,
    package_feature_dict,
    prep_dataset_and_get_iterator,
)
from tsaplay.utils._tf import masked_softmax

params = {
    "batch_size": 25,
    "max_seq_length": 85,
    "n_out_classes": 3,
    "learning_rate": 0.01,
    "location_model": 2,
    "initializer": tf.initializers.random_uniform(minval=-0.01, maxval=0.01),
    "n_hops": 8,
    "n_attn_heatmaps": 2,
}


# def memnet_input_fn(features, labels, batch_size, eval_input=False):
#     context_literals = zip_str_join(features["left"], features["right"])
#     context_mappings = zip_list_join(
#         features["mappings"]["left"], features["mappings"]["right"]
#     )

#     contexts_map, contexts_len = pad_for_dataset(context_mappings)
#     contexts = package_feature_dict(
#         mappings=contexts_map,
#         lengths=contexts_len,
#         literals=context_literals,
#         key="context",
#     )

#     target_map, target_len = pad_for_dataset(features["mappings"]["target"])
#     target_locations = [
#         len(mapping) + 1 for mapping in features["mappings"]["left"]
#     ]
#     targets = package_feature_dict(
#         mappings=target_map,
#         lengths=target_len,
#         literals=features["target"],
#         key="target",
#     )
#     targets = {**targets, "target_loc": target_locations}

#     iterator = prep_dataset_and_get_iterator(
#         features={**contexts, **targets},
#         labels=labels,
#         batch_size=batch_size,
#         eval_input=eval_input,
#     )

#     return iterator.get_next()


def memnet_pre_processing_fn(features, labels):
    processed_features = {
        "context": tf.sparse_concat(
            sp_inputs=[features["left"], features["right"]], axis=1
        ),
        "context_ids": tf.sparse_concat(
            sp_inputs=[features["left_ids"], features["right_ids"]], axis=1
        ),
        "target": features["target"],
        "target_ids": features["target_ids"],
        "target_offset": features["left"].dense_shape[1] + 1,
    }
    return processed_features, labels


def memnet_input_fn(tfrecord, batch_size, _eval=False):
    shuffle_buffer = batch_size * 10
    dataset = tf.data.TFRecordDataset(tfrecord)
    dataset = dataset.map(parse_tf_example)
    dataset = dataset.map(memnet_pre_processing_fn)

    if _eval:
        dataset = dataset.shuffle(buffer_size=shuffle_buffer)
    else:
        dataset = dataset.apply(
            tf.contrib.data.shuffle_and_repeat(buffer_size=shuffle_buffer)
        )

    dataset = dataset.batch(batch_size)

    return dataset.make_one_shot_iterator().get_next()


def memnet_serving_fn(features):
    return {
        "context_x": features["mappings"]["context"],
        "context_len": tf.add(
            features["lengths"]["left"], features["lengths"]["right"]
        ),
        "context_lit": tf.strings.join(
            [features["literals"]["left"], features["literals"]["right"]],
            separator=" ",
        ),
        "context_tok": tf.strings.join(
            [features["tok_enc"]["left"], features["tok_enc"]["right"]],
            separator="<SEP>",
        ),
        "target_x": features["mappings"]["target"],
        "target_len": features["lengths"]["target"],
        "target_lit": features["literals"]["target"],
        "target_loc": tf.add(
            features["lengths"]["left"],
            tf.ones_like(features["lengths"]["left"]),
        ),
        "target_tok": features["tok_enc"]["target"],
    }


def get_location_vector_model(model_num):
    location_vector_model = {
        1: location_vector_model_one,
        2: location_vector_model_two,
        3: location_vector_model_three,
        4: location_vector_model_four,
    }.get(model_num, 2)
    return location_vector_model


def get_absolute_distance_vector(target_locs, seq_lens, max_seq_len):
    batch_size = tf.shape(seq_lens)[0]
    mask = tf.sequence_mask(lengths=seq_lens, maxlen=max_seq_len)
    left_endings = tf.where(
        tf.equal(target_locs, 0), target_locs, target_locs - 1
    )
    locs_mask = tf.sequence_mask(lengths=left_endings, maxlen=max_seq_len)
    target_locs = tf.expand_dims(target_locs, axis=1)
    seq_range = tf.range(start=1, limit=max_seq_len + 1)
    seq_range_tiled = tf.tile([seq_range], multiples=[batch_size, 1])
    seq_range_masked = tf.where(
        locs_mask, seq_range_tiled, seq_range_tiled + 1
    )
    abs_dist = tf.abs(seq_range_masked - target_locs)

    abs_dist_masked = tf.where(mask, x=abs_dist, y=tf.cast(mask, tf.int32))

    return abs_dist_masked


def location_vector_model_one(locs, seq_lens, emb_dim, hop, init=None):
    max_seq_len = tf.shape(locs)[1]
    mask = tf.sequence_mask(lengths=seq_lens, maxlen=max_seq_len)
    seq_lens = tf.expand_dims(seq_lens, axis=1)
    seq_lens = seq_lens + tf.ones_like(seq_lens)
    v_loc = (
        1 - (locs / seq_lens) - (hop / emb_dim) * (1 - 2 * (locs / seq_lens))
    )
    v_loc = tf.cast(v_loc, tf.float32)
    v_loc = tf.where(mask, x=v_loc, y=tf.cast(mask, tf.float32))

    v_loc = tf.expand_dims(v_loc, axis=2)
    v_loc = tf.tile(v_loc, multiples=[1, 1, emb_dim])

    return v_loc


def location_vector_model_two(locs, seq_lens, emb_dim, hop=None, init=None):
    max_seq_len = tf.shape(locs)[1]
    mask = tf.sequence_mask(lengths=seq_lens, maxlen=max_seq_len)
    seq_lens = tf.expand_dims(seq_lens, axis=1)
    seq_lens = seq_lens + tf.ones_like(seq_lens)
    v_loc = 1 - (locs / seq_lens)
    v_loc = tf.cast(v_loc, tf.float32)
    v_loc = tf.where(mask, x=v_loc, y=tf.cast(mask, tf.float32))

    v_loc = tf.expand_dims(v_loc, axis=2)
    v_loc = tf.tile(v_loc, multiples=[1, 1, emb_dim])

    return v_loc


def location_vector_model_three(locs, seq_lens, emb_dim, init, hop=None):
    max_seq_len = tf.shape(locs)[1]
    max_distance = ceil(max_seq_len / 2)
    with tf.variable_scope("position_embedding_layer", reuse=tf.AUTO_REUSE):
        position_embeddings = tf.get_variable(
            name="position_embeddings",
            shape=[max_distance, emb_dim],
            dtype=tf.float32,
            initializer=init,
        )

    positional_mask = tf.sequence_mask(lengths=seq_lens, maxlen=max_seq_len)
    positional_mask = tf.expand_dims(positional_mask, axis=2)
    positional_mask = tf.tile(positional_mask, multiples=[1, 1, emb_dim])

    v_loc = tf.contrib.layers.embed_sequence(
        ids=locs,
        initializer=position_embeddings,
        scope="pos_emb_layer",
        reuse=True,
    )

    v_loc = tf.where(
        positional_mask, v_loc, tf.cast(positional_mask, tf.float32)
    )

    return v_loc


def location_vector_model_four(locs, seq_lens, emb_dim, init, hop=None):
    v_loc_three = location_vector_model_three(locs, seq_lens, emb_dim, init)
    max_seq_len = tf.shape(locs)[1]

    positional_mask = tf.sequence_mask(lengths=seq_lens, maxlen=max_seq_len)
    positional_mask = tf.expand_dims(positional_mask, axis=2)
    positional_mask = tf.tile(positional_mask, multiples=[1, 1, emb_dim])

    v_loc = tf.nn.sigmoid(v_loc_three)

    v_loc = tf.where(
        positional_mask, v_loc, tf.cast(positional_mask, tf.float32)
    )

    return v_loc


def memnet_content_attn_unit(seq_lens, memory, v_aspect, emb_dim, init):
    batch_size = tf.shape(memory)[0]
    max_seq_len = tf.shape(memory)[1]
    w_att = tf.get_variable(
        name="weights",
        shape=[1, 2 * emb_dim],
        dtype=tf.float32,
        initializer=init,
    )
    b_att = tf.get_variable(
        name="bias", shape=[1], dtype=tf.float32, initializer=init
    )

    w_att_batch_dim = tf.expand_dims(w_att, axis=0)
    w_att_tiled = tf.tile(
        w_att_batch_dim, multiples=[batch_size * max_seq_len, 1, 1]
    )
    w_att = tf.reshape(w_att_tiled, shape=[-1, max_seq_len, 1, 2 * emb_dim])

    b_att_mask = tf.sequence_mask(
        lengths=seq_lens, maxlen=max_seq_len, dtype=tf.float32
    )
    b_att = b_att_mask * b_att
    b_att = tf.reshape(b_att, shape=[batch_size, -1, 1, 1])

    v_aspect_tiled = tf.tile(v_aspect, multiples=[1, max_seq_len, 1])

    v_aspect_mask = tf.sequence_mask(lengths=seq_lens, maxlen=max_seq_len)
    v_aspect_mask = tf.expand_dims(v_aspect_mask, axis=2)
    v_aspect_mask = tf.tile(v_aspect_mask, multiples=[1, 1, emb_dim])

    v_aspect_tiled = tf.where(
        v_aspect_mask, v_aspect_tiled, tf.cast(v_aspect_mask, tf.float32)
    )

    mem_v_aspect = tf.concat([memory, v_aspect_tiled], axis=2)
    mem_v_aspect = tf.expand_dims(mem_v_aspect, axis=3)

    g_score = tf.nn.tanh(
        tf.einsum("Baij,Bajk->Baik", w_att, mem_v_aspect) + b_att
    )
    g_score = tf.squeeze(g_score, axis=3)

    softmax_mask = tf.sequence_mask(lengths=seq_lens, maxlen=max_seq_len)

    attn_vec = masked_softmax(logits=g_score, mask=softmax_mask)

    output_vec = tf.reduce_sum(memory * attn_vec, axis=1)

    return (
        output_vec,  # dim: [batch_size, embedding_dim]
        attn_vec,  # to optionally use for summary heatmaps
    )
