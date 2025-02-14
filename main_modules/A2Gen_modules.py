import numpy as np
import tensorflow as tf


class ContextAwareAttention:
    def __init__(self, emb_size, num_heads, name):
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.name = name

    def self_attention(self, queries, keys, values, mask=None):
        Q = tf.expand_dims(tf.layers.dense(queries, self.emb_size), axis=-3)
        K = tf.expand_dims(tf.layers.dense(keys, self.emb_size), axis=-3)
        V = tf.expand_dims(tf.layers.dense(values, self.emb_size), axis=-3)

        Q_ = tf.concat(tf.split(Q, self.num_heads, axis=-1), axis=-3)
        K_ = tf.concat(tf.split(K, self.num_heads, axis=-1), axis=-3)
        V_ = tf.concat(tf.split(V, self.num_heads, axis=-1), axis=-3)

        scores = tf.matmul(Q_, tf.transpose(K_, [0, 1, 2, 4, 3])) / np.sqrt(self.emb_size // self.num_heads)
        if mask is not None:
            attention_weights = tf.nn.softmax(scores + mask, axis=-1)
        else:
            attention_weights = tf.nn.softmax(scores, axis=-1)

        output = tf.matmul(attention_weights, V_)
        output = tf.layers.dense(output, self.emb_size)
        return output

    def gate(self, context_features):
        gate_weights = tf.layers.dense(context_features, self.num_heads, activation=tf.nn.softmax)
        return gate_weights

    def cam(self, seq_features, context_features, mask=None):
        with tf.variable_scope(f"cam_{self.name}_layer", reuse=tf.AUTO_REUSE):
            seq_f = tf.layers.dense(seq_features, self.emb_size)
            ctx_f = tf.layers.dense(context_features, self.emb_size)
            ctx_f = tf.tile(ctx_f[:, tf.newaxis, :], [1, tf.shape(seq_f)[1], 1])

            seq_f = self.self_attention(tf.concat([seq_f, ctx_f]), seq_f, seq_f, mask)
            attention_x = seq_f * tf.expand_dims(tf.expand_dims(self.gate(context_features)), axis=1)
            attention_x = tf.reduce_sum(attention_x, axis=2)
        return attention_x + ctx_f


class HierarchicalSequenceEncoder:
    def __init__(self, emb_size, num_heads):
        self.act_cam = ContextAwareAttention(emb_size, num_heads, "hse_act_cam")
        self.item_cam = ContextAwareAttention(emb_size, num_heads, "hse_item_cam")

    def hse(self, action_seq, item_seq, target_item):
        with tf.variable_scope(f"hse_layer", reuse=tf.AUTO_REUSE):
            act_cam_output = tf.reduce_sum(self.act_cam.cam(action_seq, item_seq), axis=-2)
            item_cam_output = tf.reduce_sum(self.item_cam.cam(tf.concat([item_seq, act_cam_output]), target_item), axis=-2)
            return item_cam_output


class ExplicitAutoregressiveGenerator:
    def __init__(self, emb_size, num_heads, action_type_num):
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.action_type_num = action_type_num

        self.hse = HierarchicalSequenceEncoder(emb_size, num_heads)
        self.gen_cam = ContextAwareAttention(emb_size, num_heads, "eag_cam")

    def eag(self,
            hist_action_seq,
            hist_item_seq,
            target_action_seq,
            target_item,
            user_emb
            ):
        with tf.variable_scope(f"eag_layer", reuse=tf.AUTO_REUSE):
            mask = tf.fill([1, 1, self.action_type_num, self.action_type_num], float('-inf'))
            for i in range(self.action_type_num):
                for j in range(i + 1):
                    mask = tf.tensor_scatter_nd_update(mask, [[0, 0, i, j]], [0])

            hse_output = self.hse.hse(hist_action_seq, hist_item_seq, target_item)
            ctx_f = tf.concat([user_emb, hse_output], axis=-1)
            seq_output = self.gen_cam.cam(target_action_seq, ctx_f, mask)
            time_pred = tf.layers.dense(seq_output, 1)
            cls_pred_logits = tf.layers.dense(seq_output, self.action_type_num + 1, activation=tf.nn.softmax)

        return time_pred, cls_pred_logits


class ImplicitAutoregressiveGenerator:
    def __init__(self, emb_size, num_heads, action_type_num):
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.action_type_num = action_type_num

        self.hse = HierarchicalSequenceEncoder(emb_size, num_heads)
        self.gen_cam = ContextAwareAttention(emb_size, num_heads, "iag_cam")

    def auxiliary_pred(self, x):
        time_pred_aux = tf.layers.dense(x, 1)
        cls_pred_logits_aux = tf.layers.dense(x, self.action_type_num + 1)
        return time_pred_aux, cls_pred_logits_aux

    def iag(self,
            hist_action_seq,
            hist_item_seq,
            target_item,
            user_emb
            ):
        with tf.variable_scope(f"iag_layer", reuse=tf.AUTO_REUSE):
            hse_output = self.hse.hse(hist_action_seq, hist_item_seq, target_item)
            ctx_f = tf.tile(
                tf.concat([user_emb, hse_output], axis=-1)[:, tf.newaxis, :],
                [1, self.action_type_num, 1]
            )
            pred_logits_seq = tf.layers.dense(ctx_f, self.emb_size)
            time_pred_aux, cls_pred_logits_aux = self.auxiliary_pred(pred_logits_seq)

            seq_output = self.gen_cam.cam(pred_logits_seq, ctx_f)
            time_pred = tf.layers.dense(seq_output, 1)
            cls_pred_logits = tf.layers.dense(seq_output, self.action_type_num + 1)

        return time_pred_aux, cls_pred_logits_aux, time_pred, cls_pred_logits
