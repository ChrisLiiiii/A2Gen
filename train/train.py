import tensorflow as tf
import kuaishou_tf as k
from main_modules.A2Gen_modules import ExplicitAutoregressiveGenerator, ImplicitAutoregressiveGenerator
from data_loader.data_loader import data_loader, action_types
from config.args import args


hist_features, main_features, labels = data_loader()

hist_item_emb_seq, hist_action_time_emb_seq = hist_features
user_emb, target_item_emb = main_features
label_action_seq, label_time_seq, label_action_time_emb_seq = labels

model_type = "EAG"  # "EAG" or "IAG"

step = 0
print_tensor_opt = []
with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
    print("model_start")
    # Hyperparameters
    emb_size = 32
    num_heads = 4
    action_type_num = len(action_types) - 1  # the num of action types to predict, remove start

    if model_type == "EAG":
        A2Gen = ExplicitAutoregressiveGenerator(
            emb_size=emb_size,
            num_heads=num_heads,
            action_type_num=action_type_num
        )
        time_pred, cls_pred_logits = A2Gen.eag(
            hist_action_time_emb_seq,
            hist_item_emb_seq,
            label_action_seq,
            target_item_emb,
            user_emb
        )
    elif model_type == "IAG":
        A2Gen = ImplicitAutoregressiveGenerator(
            emb_size=emb_size,
            num_heads=num_heads,
            action_type_num=action_type_num
        )
        time_pred_aux, cls_pred_logits_aux, time_pred, cls_pred_logits = A2Gen.iag(
            hist_action_time_emb_seq,
            hist_item_emb_seq,
            target_item_emb,
            user_emb
        )
    else:
        raise ValueError("model_type must be 'EAG' or 'IAG'")

if args.mode == 'train':
    loss_list = []

    ## ---------------------------- softmax loss ----------------------------
    label_action_seq = tf.cast(label_action_seq[:, 1:], tf.int32) - 1  # remove start
    label_one_hot_seq = tf.one_hot(label_action_seq, axis=-1, depth=len(action_types) - 1)

    # padding mask loss
    padding_mask = tf.cast(tf.math.logical_not(tf.math.equal(label_action_seq, len(action_types) - 2)), dtype=tf.float32)
    softmax_loss = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=label_one_hot_seq,
            logits=cls_pred_logits,
            axis=-1
        ) * padding_mask
    softmax_loss = tf.reduce_sum(softmax_loss, axis=-1)
    loss_list.append((
        f"softmax_loss",
        softmax_loss,
        1
    ))
    valid_labels = tf.boolean_mask(label_time_seq, padding_mask)
    valid_preds = tf.boolean_mask(time_pred, padding_mask)
    mse_loss = tf.losses.mean_squared_error(valid_labels, valid_preds)
    loss_list.append((
        f"mse_loss",
        mse_loss,
        0.1
    ))
    if model_type == "IAG":
        softmax_loss_aux = tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=label_one_hot_seq,
                logits=cls_pred_logits_aux,
                axis=-1
            ) * padding_mask
        softmax_loss_aux = tf.reduce_sum(softmax_loss_aux, axis=-1)
        loss_list.append((
            f"softmax_loss_aux",
            softmax_loss_aux,
            0.1
        ))
        valid_preds_aux = tf.boolean_mask(time_pred_aux, padding_mask)
        mse_loss_aux = tf.losses.mean_squared_error(valid_labels, valid_preds_aux)
        loss_list.append((
            f"mse_loss",
            mse_loss_aux,
            0.1
        ))

    # order loss
    rolled_time_pred = tf.roll(time_pred, shift=1, axis=1)
    rolled_time_pred = tf.concat(
        [tf.tile(tf.constant(float('-inf'))[tf.newaxis, tf.newaxis], [tf.shape(time_pred)[0], 1]), rolled_time_pred[:, 1:]],
        axis=1
    )
    action_type_pred = tf.nn.softmax(cls_pred_logits, axis=-1)[:, :, :-1]  # remove padding
    max_indices = tf.argmax(action_type_pred, axis=-1)  # -1, 9
    is_leave = tf.cast(tf.equal(max_indices, len(action_types) - 3), dtype=tf.int32)  # find where are leave actions
    first_exit_index = tf.argmax(is_leave, axis=-1, output_type=tf.int32)[:, tf.newaxis]
    row_indices = tf.range(len(action_types) - 2, dtype=tf.int32)[tf.newaxis, :]
    mask_leave = tf.expand_dims(tf.cast(tf.less(row_indices, first_exit_index + 1), tf.float32), axis=-1)  # mask after leave actions
    time_differences = (time_pred - rolled_time_pred) * tf.squeeze(mask_leave, axis=-1)
    loss_list.append((
        f"time_order_loss",
        tf.reduce_sum(tf.maximum(-time_differences, 0), axis=-1),
        0.01
    ))

    total_loss = 0.0
    for loss_name, loss, loss_weight in loss_list:
        total_loss += loss * loss_weight

        sparse_optimizer = k.config.optimizer.Adam(0.0001)
        dense_optimizer = k.config.optimizer.Adam(0.0005)
        sparse_optimizer.minimize(total_loss, var_list=k.config.get_collection(k.config.GraphKeys.EMBEDDING_INPUT))
        dense_optimizer.minimize(total_loss, var_list=k.config.get_collection(k.config.GraphKeys.TRAINABLE_VARIABLES))
