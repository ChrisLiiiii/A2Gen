import tensorflow as tf
from config.args import args


def mlp_block(name, inputs, units, dropouts=None, activation=tf.nn.leaky_relu, stop_gradient=False):
    with tf.variable_scope(f'{name}_mlp_tower', reuse=tf.AUTO_REUSE):
        if stop_gradient:
            output = tf.stop_gradient(inputs)
        else:
            output = inputs
        for i in range(len(units)):
            output = tf.layers.dense(output, units[i], activation=activation,
                                     kernel_initializer=tf.glorot_uniform_initializer())
            if dropouts is not None and dropouts[i] > 0:
                output = tf.layers.dropout(output, dropouts[i], training=(args.mode == 'train'))
        return output

