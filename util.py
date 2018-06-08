"""Helper graphs for building the network."""

__author__ = 'Connor Sanchez'

import math
import tensorflow as tf


def conv3x3_x2_relu(inputs, in_filters, out_filters):
    conv1 = tf.layers.conv2d(
        inputs=inputs,
        filters=out_filters,
        kernel_size=[3, 3],
        padding='same',
        activation=tf.nn.relu,
        kernel_initializer=tf.initializers.truncated_normal(
            stddev=math.sqrt(2 / (3 * 3 * in_filters))
        )
    )
    conv2 = tf.layers.conv2d(
        inputs=conv1,
        filters=out_filters,
        kernel_size=[3, 3],
        padding='same',
        activation=tf.nn.relu,
        kernel_initializer=tf.initializers.truncated_normal(
            stddev=math.sqrt(2 / (3 * 3 * out_filters))
        )
    )
    return conv2


def conv1x1_relu(inputs, in_filters, out_filters):
    conv = tf.layers.conv2d(
        inputs=inputs,
        filters=out_filters,
        kernel_size=[1, 1],
        padding='same',
        activation=tf.nn.relu,
        kernel_initializer=tf.initializers.truncated_normal(
            stddev=math.sqrt(2 / (1 * 1 * in_filters))
        )
    )
    return conv


def max_pool2x2(inputs):
    down = tf.layers.max_pooling2d(
        inputs=inputs,
        pool_size=[2, 2],
        strides=2
    )
    return down


def conv2x2_transpose(inputs, in_filters):
    up = tf.layers.conv2d_transpose(
        inputs=inputs,
        filters=in_filters // 2,
        kernel_size=[2, 2],
        strides=2,
        padding='same',
        kernel_initializer=tf.initializers.truncated_normal(
            stddev=math.sqrt(2 / (2 * 2 * in_filters))
        )
    )
    return up


def concat_depth(inputs_left, inputs_right):
    concat = tf.concat(
        values=[inputs_left, inputs_right],
        axis=-1
    )
    return concat


def down_and_conv(inputs, in_filters, out_filters):
    down = max_pool2x2(inputs)
    conv = conv3x3_x2_relu(down, in_filters, out_filters)
    return conv


def up_and_concat_and_conv(inputs, inputs_left, in_filters, out_filters):
    up = conv2x2_transpose(inputs, in_filters)
    shape_up = up.get_shape().as_list()
    shape_left = inputs_left.get_shape().as_list()
    diff1 = shape_left[1] - shape_up[1]
    diff2 = shape_left[2] - shape_up[2]
    up_padded = tf.pad(up, paddings=[[0, 0], [0, diff1], [0, diff2], [0, 0]], mode='SYMMETRIC')
    concat = concat_depth(inputs_left, up_padded)
    conv = conv3x3_x2_relu(concat, in_filters, out_filters)
    return conv


if __name__ == '__main__':
    print('util')
