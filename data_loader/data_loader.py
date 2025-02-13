import tensorflow as tf
import kuaishou_tf as k
from main_modules.modules import mlp_block
from config.args import args

# ------------------------------------ feature config ------------------------------------
hist_item_num = args.history_length
emb_size = args.embedding_size

action_types = ["start", "like", "follow", "forward", "collect", "exit", "padding"]
action_emb_dict = tf.Variable(tf.random_normal([len(action_types), emb_size]), name='action_emb_dict')


def data_loader():
    # get feature emb from kafka data stream
    user_emb = k.nn.new_embedding("user_emb", dim=emb_size)
    target_item_emb = k.nn.new_embedding("target_pid_emb", dim=emb_size)

    # ------------------------------------ extract history item seq ------------------------------------
    hist_item_emb_seq = k.nn.new_embedding('hist_item_emb_seq', dim=emb_size, expand=hist_item_num)
    hist_item_emb_seq = tf.reshape(hist_item_emb_seq, [-1, hist_item_num, emb_size])  # ?, 10, 32

    time_action_emb_list = []  # (?, 10, 1, 33) * 6
    for i, action_type in enumerate(action_types[:-1]):  # except padding
        is_act = tf.reshape(  # (?, 10, 1)
            k.nn.get_dense_fea(f"is_{action_type}_list", dim=hist_item_num, dtype=tf.float32),
            [-1, hist_item_num, 1])
        act_time_emb = tf.reshape(  # (?, 10, 32)
            k.nn.get_dense_fea(f"{action_type}_time_emb_list", dim=hist_item_num, dtype=tf.float32),
            [-1, hist_item_num, emb_size])

        action_emb = action_emb_dict[i, :][tf.newaxis, tf.newaxis, :]  # (1, 1, 32)
        action_emb = tf.tile(  # (?, 10, 32)
            action_emb,
            [tf.shape(is_act)[0], hist_item_num, 1]
        )
        time_action_emb = tf.concat([act_time_emb, action_emb], axis=-1)  # (?, 10, 64)
        time_action_emb_list.append(tf.expand_dims(is_act * time_action_emb, axis=2))  # (?, 10, 1, 64)

    hist_action_time_emb_seq = tf.concat(time_action_emb_list, axis=2)  # (?, 10, 9, 64)

    # other user hist item seq (e.g., like item seq)
    user_other_seq_list = []
    other_seq_length = 20
    for i in range(5):
        user_seq_embed = k.nn.new_embedding(f'user_list_{i}', dim=emb_size, expand=other_seq_length)
        user_seq_embed = tf.reshape(user_seq_embed, [-1, other_seq_length, emb_size])  # ?, 20, 32
        user_other_seq_list.append(user_seq_embed)
    user_other_seq = tf.reduce_mean(tf.concat(user_other_seq_list, axis=2), axis=1)

    # ------------------------ label construct ------------------------ #
    label_action_seq = k.nn.get_dense_fea(f"action_seq", dim=len(action_types) - 1, dtype=tf.int32)  # -1, 10
    label_time_seq = k.nn.get_dense_fea(f"time_seq", dim=len(action_types) - 1, dtype=tf.float32)  # -1, 10
    label_time_emb_seq = k.nn.new_embedding(f"time_emb_seq", dim=len(action_types) - 1)
    label_time_emb_seq = tf.reshape(label_time_emb_seq, [-1, len(action_types), emb_size])  # -1, 10, 32
    label_action_type_onehot_seq = tf.one_hot(label_action_seq, axis=-1, depth=len(action_types))  # -1, 9, 11
    label_action_emb_seq = tf.matmul(label_action_type_onehot_seq, action_emb_dict)  # -1, 9, 32

    label_action_time_emb_seq = tf.concat([tf.expand_dims(label_time_emb_seq, axis=-1), label_action_emb_seq], axis=-1)  # -1, 9, 64
    # ------------------------------------------------------------------ #

    user_other_seq_output = mlp_block('seq_process', user_other_seq, [emb_size])
    user_emb = tf.concat([user_emb, user_other_seq_output], axis=-1)  # (?, 128)

    return (hist_item_emb_seq, hist_action_time_emb_seq), (user_emb, target_item_emb), (label_action_seq, label_time_seq, label_action_time_emb_seq)

