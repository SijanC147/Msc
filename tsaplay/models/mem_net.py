from math import ceil
import tensorflow as tf
from tsaplay.models.tsa_model import TsaModel
from tsaplay.utils.addons import addon, attn_heatmaps
from tsaplay.utils.tf import (
    masked_softmax,
    variable_len_batch_mean,
    generate_attn_heatmap_summary,
    append_snapshot,
    create_snapshots_container,
    zip_attn_snapshots_with_literals,
)


class MemNet(TsaModel):
    def set_params(self):
        return {
            ### Taken from https://github.com/NUSTM/ABSC/blob/master/models/ABSC_Zozoz/model/dmn.py ###
            "batch-size": 100,
            ###
            "learning_rate": 0.01,
            "location_model": 2,
            "n_hops": 9,
            "initializer": tf.initializers.random_uniform(-0.01, 0.01),
            # From original paper, "... we clamp the word embeddings ..."
            "train_embeddings": False,
        }

    @classmethod
    def processing_fn(cls, features):
        return {
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

    @addon([attn_heatmaps])
    def model_fn(self, features, labels, mode, params):
        target_offset = tf.cast(features["target_offset"], tf.int32)

        memory = features["context_emb"]

        max_ctxt_len = tf.shape(memory)[1]

        context_locations = get_absolute_distance_vector(
            target_locs=target_offset,
            seq_lens=features["context_len"],
            max_seq_len=max_ctxt_len,
        )

        v_aspect = variable_len_batch_mean(
            input_tensor=features["target_emb"],
            seq_lengths=features["target_len"],
            op_name="target_embedding_avg",
        )

        attn_snapshots = create_snapshots_container(
            shape_like=features["context_ids"], n_snaps=params["n_hops"]
        )

        hop_number = tf.constant(1)

        initial_hop_inputs = (hop_number, memory, v_aspect, attn_snapshots)

        def condition(hop_num, ext_memory, input_vec, attn_snapshots):
            return tf.less_equal(hop_num, params["n_hops"])

        def hop(hop_num, ext_memory, input_vec, attn_snapshots):
            location_vector_model_fn = get_location_vector_model(
                model_num=params["location_model"]
            )

            v_loc = location_vector_model_fn(
                locs=context_locations,
                seq_lens=features["context_len"],
                emb_dim=params["_embedding_dim"],
                hop=hop_num,
                init=params["initializer"],
            )

            if params["location_model"] == 3:
                ext_memory = ext_memory + v_loc
            else:
                ext_memory = tf.multiply(memory, v_loc)

            with tf.variable_scope("linear_layer", reuse=tf.AUTO_REUSE):
                linear_out = tf.layers.dense(
                    inputs=tf.squeeze(input_vec, axis=1),
                    units=params["_embedding_dim"],
                    activation=None,
                    kernel_initializer=params["initializer"],
                    bias_initializer=params["initializer"],
                )

            with tf.variable_scope("attention_layer", reuse=tf.AUTO_REUSE):
                attn_out, attn_snapshot = memnet_content_attn_unit(
                    seq_lens=features["context_len"],
                    memory=ext_memory,
                    v_aspect=input_vec,
                    emb_dim=params["_embedding_dim"],
                    init=params["initializer"],
                )

            attn_snapshots = append_snapshot(
                container=attn_snapshots, new_snap=attn_snapshot, index=hop_num
            )

            output_vec = attn_out + linear_out
            output_vec = tf.expand_dims(output_vec, axis=1)

            hop_num = tf.add(hop_num, 1)

            return (hop_num, ext_memory, output_vec, attn_snapshots)

        _, _, final_sentence_rep, attn_snapshots = tf.while_loop(
            cond=condition,
            body=hop,
            loop_vars=initial_hop_inputs,
            shape_invariants=(
                hop_number.get_shape(),
                memory.get_shape(),
                v_aspect.get_shape(),
                tf.TensorShape(dims=[params["n_hops"], None, None, 1]),
            ),
        )

        literals, attn_snapshots = zip_attn_snapshots_with_literals(
            literals=features["context"],
            snapshots=attn_snapshots,
            num_layers=params["n_hops"],
        )
        attn_info = tf.tuple([literals, attn_snapshots])
        generate_attn_heatmap_summary(attn_info)

        final_sentence_rep = tf.squeeze(final_sentence_rep, axis=1)

        logits = tf.layers.dense(
            inputs=final_sentence_rep,
            units=params["_n_out_classes"],
            kernel_initializer=params["initializer"],
            bias_initializer=params["initializer"],
        )

        loss = tf.losses.sparse_softmax_cross_entropy(
            labels=labels, logits=logits
        )

        optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=params["learning_rate"]
        )

        return self.make_estimator_spec(
            mode=mode, logits=logits, optimizer=optimizer, loss=loss
        )


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


def get_location_vector_model(model_num):
    location_vector_model = {
        1: location_vector_model_one,
        2: location_vector_model_two,
        3: location_vector_model_three,
        4: location_vector_model_four,
    }.get(model_num, 2)
    return location_vector_model


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


def memnet_content_attn_unit(
    seq_lens, memory, v_aspect, emb_dim, init, bias_init=None
):
    batch_size = tf.shape(memory)[0]
    max_seq_len = tf.shape(memory)[1]
    w_att = tf.get_variable(
        name="weights",
        shape=[1, 2 * emb_dim],
        dtype=tf.float32,
        initializer=init,
    )
    b_att = tf.get_variable(
        name="bias",
        shape=[1],
        dtype=tf.float32,
        initializer=(bias_init or init),
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
