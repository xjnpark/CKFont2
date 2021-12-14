import tensorflow as tf
import tensorflow.contrib as tf_contrib

from utils import pytorch_kaiming_weight_factor
import numpy as np

##################################################################################
# Initialization
##################################################################################

factor, mode, uniform = pytorch_kaiming_weight_factor(a=0.0, uniform=False)
weight_init = tf_contrib.layers.variance_scaling_initializer(factor=factor, mode=mode, uniform=uniform)
# weight_init = tf.random_normal_initializer(mean=0.0, stddev=0.02)

weight_regularizer = tf.contrib.layers.l2_regularizer(0.0001)
weight_regularizer_fully = tf.contrib.layers.l2_regularizer(0.0001)

##################################################################################
# CNN operations
##################################################################################
def discrim_conv(batch_input, out_channels, stride):
    padded_input = tf.pad(batch_input, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
    return tf.layers.conv2d(padded_input, out_channels, kernel_size=4, strides=(stride, stride), padding="valid", kernel_initializer=tf.random_normal_initializer(0, 0.02))


def gen_conv(batch_input, out_channels, args):
    # [batch, in_height, in_width, in_channels] => [batch, out_height, out_width, out_channels]
    initializer = tf.random_normal_initializer(0, 0.02)
    if args.separable_conv:
        return tf.layers.separable_conv2d(batch_input, out_channels, kernel_size=4, strides=(2, 2), padding="same", depthwise_initializer=initializer, pointwise_initializer=initializer)
    else:
        return tf.layers.conv2d(batch_input, out_channels, kernel_size=4, strides=(2, 2), padding="same", kernel_initializer=initializer)

def gen_deconv(batch_input, out_channels, args):
    # [batch, in_height, in_width, in_channels] => [batch, out_height, out_width, out_channels]
    initializer = tf.random_normal_initializer(0, 0.02)
    if args.separable_conv:
        _b, h, w, _c = batch_input.shape
        resized_input = tf.image.resize_images(batch_input, [h * 2, w * 2], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        return tf.layers.separable_conv2d(resized_input, out_channels, kernel_size=4, strides=(1, 1), padding="same", depthwise_initializer=initializer, pointwise_initializer=initializer)
    else:
        return tf.layers.conv2d_transpose(batch_input, out_channels, kernel_size=4, strides=(2, 2), padding="same", kernel_initializer=initializer)


def lrelu(x, a):
    with tf.name_scope("lrelu"):
        # adding these together creates the leak part and linear part
        # then cancels them out by subtracting/adding an absolute value term
        # leak: a*x/2 - a*abs(x)/2
        # linear: x/2 + abs(x)/2

        # this block looks like it has 2 inputs on the graph unless we do this
        x = tf.identity(x)
        return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)


def batchnorm(inputs):
    return tf.layers.batch_normalization(inputs, axis=3, epsilon=1e-5, momentum=0.1, training=True, gamma_initializer=tf.random_normal_initializer(1.0, 0.02))

def instance_norm(x):
    return tf_contrib.layers.instance_norm(x, epsilon=1e-05, center=True, scale=True)

##################################################################################
# New OPS for New architecture
##################################################################################

def resblock(x_init, channels, use_bias=True, sn=False):
    x = conv(x_init, channels, kernel=3, stride=1, pad=1, use_bias=use_bias, sn=sn)
    x = instance_norm(x)
    x = tf.nn.relu(x)

    x = conv(x, channels, kernel=3, stride=1, pad=1, use_bias=use_bias, sn=sn)
    x = instance_norm(x)

    return x + x_init


def conv(x, channels, kernel=4, stride=2, pad=0, pad_type='zero', use_bias=True, sn=False):
    if pad > 0:
        h = x.get_shape().as_list()[1]
        if h % stride == 0:
            pad = pad * 2
        else:
            pad = max(kernel - (h % stride), 0)

        pad_top = pad // 2
        pad_bottom = pad - pad_top
        pad_left = pad // 2
        pad_right = pad - pad_left

        if pad_type == 'zero':
            x = tf.pad(x, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]])
        if pad_type == 'reflect':
            x = tf.pad(x, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], mode='REFLECT')

    if sn:
        print("Spectral Normailzation is True")

    else:
        x = tf.layers.conv2d(inputs=x, filters=channels,
                             kernel_size=kernel, kernel_initializer=weight_init,
                             kernel_regularizer=weight_regularizer,
                             strides=stride, use_bias=use_bias)

    return x

def pre_resblock(x_init, channels, use_bias=True, sn=False):
    _, _, _, init_channel = x_init.get_shape().as_list()

    x = lrelu(x_init, 0.2)
    x = conv(x, channels, kernel=3, stride=1, pad=1, use_bias=use_bias, sn=sn)

    x = lrelu(x, 0.2)
    x = conv(x, channels, kernel=3, stride=1, pad=1, use_bias=use_bias, sn=sn)

    if init_channel != channels:
        x_init = conv(x_init, channels, kernel=1, stride=1, use_bias=False, sn=sn)

    return x + x_init

##################################################################################
# Sampling
##################################################################################

def up_sample(x, scale_factor=2):
    _, h, w, _ = x.get_shape().as_list()
    new_size = [h * scale_factor, w * scale_factor]
    return tf.image.resize_nearest_neighbor(x, size=new_size)

def global_avg_pooling(x):
    gap = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
    return gap

def down_sample_avg(x, scale_factor=2):
    return tf.layers.average_pooling2d(x, pool_size=3, strides=scale_factor, padding='SAME')