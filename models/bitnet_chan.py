import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim


def init():
    def _initializer(shape, dtype=tf.float32, partition_info=None):
        array = np.zeros(shape)
        cx, cy = shape[0]//2, shape[1]//2
        for i in range(np.minimum(shape[2], shape[3])):
            array[cx, cy, i, i] = 1
        return tf.constant(array, dtype=dtype)
    return _initializer


def conv2d(x, chan, kernel, stride, rate, name, flag='act'):
    if flag == '1x1':
        return slim.conv2d(tf.nn.leaky_relu(x), chan, kernel, activation_fn=None, scope=name+'_1x1_c')
    elif flag == 'no_act':
        return slim.conv2d(x, chan, kernel, stride, rate=rate, activation_fn=None, weights_initializer=init(), scope=name+'_no_act_c')
    else:
        return slim.conv2d(tf.nn.leaky_relu(x), chan, kernel, stride, rate=rate, activation_fn=None, weights_initializer=init(), scope=name+'_c')


def trans_conv2d(x, chan, kernel, stride, name, output_shape):
    x_shape = x.get_shape().as_list()
    kernel = tf.get_variable(initializer=init(), shape=[kernel, kernel, chan, x_shape[3]], name=name+'_k')
    return tf.nn.conv2d_transpose(tf.nn.leaky_relu(x), kernel, output_shape, [1, stride, stride, 1], name=name)


def enc_dec(x, name):
    # Down-sampling
    d1 = conv2d(conv2d(x,  32,  3, 2, 1, name+'_d1_1'), 32,  3, 1, 2, name+'_d1_2')
    d2 = conv2d(conv2d(d1, 32,  3, 2, 1, name+'_d2_1'), 32,  3, 1, 2, name+'_d2_2')
    d3 = conv2d(conv2d(d2, 64,  3, 2, 1, name+'_d3_1'), 64,  3, 1, 2, name+'_d3_2')
    d4 = conv2d(conv2d(d3, 64,  3, 2, 1, name+'_d4_1'), 64,  3, 1, 2, name+'_d4_2')
    d5 = conv2d(conv2d(d4, 128, 3, 2, 1, name+'_d5_1'), 128, 3, 1, 2, name+'_d5_2')

    # Middle stage
    mid = conv2d(d5, 128, 3, 1, 2, name+'_mid')

    # Up-sampling, connection
    u1 = trans_conv2d(conv2d(mid, 128, 3, 1, 2, name+'_u1_1'), 64, 3, 2, name+'_u1_2', tf.shape(d4)) + d4
    u2 = trans_conv2d(conv2d(u1,  64,  3, 1, 2, name+'_u2_1'), 64, 3, 2, name+'_u2_2', tf.shape(d3)) + d3
    u3 = trans_conv2d(conv2d(u2,  64,  3, 1, 2, name+'_u3_1'), 32, 3, 2, name+'_u3_2', tf.shape(d2)) + d2
    u4 = trans_conv2d(conv2d(u3,  32,  3, 1, 2, name+'_u4_1'), 32, 3, 2, name+'_u4_2', tf.shape(d1)) + d1
    u5 = trans_conv2d(conv2d(u4,  32,  3, 1, 2, name+'_u5_1'), 32, 3, 2, name+'_u5_2', tf.shape(x))  + x

    # Final aggregation
    result = conv2d(u5, 32, 1, 1, 1, name+'_f1', '1x1') \
             + conv2d(tf.image.resize_bilinear(u4,  tf.shape(x)[1:3]), 32, 1, 1, 1, name+'_f2', '1x1') \
             + conv2d(tf.image.resize_bilinear(u3,  tf.shape(x)[1:3]), 32, 1, 1, 1, name+'_f3', '1x1') \
             + conv2d(tf.image.resize_bilinear(u2,  tf.shape(x)[1:3]), 32, 1, 1, 1, name+'_f4', '1x1') \
             + conv2d(tf.image.resize_bilinear(u1,  tf.shape(x)[1:3]), 32, 1, 1, 1, name+'_f5', '1x1') \
             + conv2d(tf.image.resize_bilinear(mid, tf.shape(x)[1:3]), 32, 1, 1, 1, name+'_f6', '1x1') \

    return result


def net(x, reuse):
    with tf.variable_scope('enc_dec_v4', reuse=reuse):
        # First convolution
        c1 = conv2d(x,   32, 3, 1, 1, 'c1', 'no_act')
        c2 = conv2d(c1,  32, 3, 1, 1, 'c2')

        # Encoder - decoder, infer
        infer = enc_dec(c2, 'enc_dec')
        infer = conv2d(infer, 32, 3, 1, 1, 'infer1')
        infer = conv2d(infer, 1,  1, 1, 1, 'infer2', '1x1')

    return infer
