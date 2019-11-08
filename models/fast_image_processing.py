import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim


def lrelu(x):
    return tf.maximum(x*0.2,x)


def identity_initializer():
    def _initializer(shape, dtype=tf.float32, partition_info=None):
        array = np.zeros(shape, dtype=float)
        cx, cy = shape[0]//2, shape[1]//2
        for i in range(np.minimum(shape[2],shape[3])):
            array[cx, cy, i, i] = 1
        return tf.constant(array, dtype=dtype)
    return _initializer


def nm(x):
    # The parameter "is_training" in slim.batch_norm does not seem to help so I do not use it
    w0 = tf.get_variable('w0', shape=[],  initializer=tf.constant_initializer(1.0))
    w1 = tf.get_variable('w1', shape=[],  initializer=tf.constant_initializer(0.0))
    return w0 * x + w1 * slim.batch_norm(x)


def net(x, reuse):
    with tf.variable_scope('fast_image_processing', reuse=reuse):
        net = slim.conv2d(x, 32, [3, 3], rate=1, activation_fn=lrelu, normalizer_fn=nm,
                          weights_initializer=identity_initializer(), scope='g_conv1')
        net = slim.conv2d(net, 32, [3, 3], rate=2, activation_fn=lrelu, normalizer_fn=nm,
                          weights_initializer=identity_initializer(), scope='g_conv2')
        net = slim.conv2d(net, 32, [3, 3], rate=4, activation_fn=lrelu, normalizer_fn=nm,
                          weights_initializer=identity_initializer(), scope='g_conv3')
        net = slim.conv2d(net, 32, [3, 3], rate=8, activation_fn=lrelu, normalizer_fn=nm,
                          weights_initializer=identity_initializer(), scope='g_conv4')
        net = slim.conv2d(net, 32, [3, 3], rate=16, activation_fn=lrelu, normalizer_fn=nm,
                          weights_initializer=identity_initializer(), scope='g_conv5')
        net = slim.conv2d(net, 32, [3, 3], rate=32, activation_fn=lrelu, normalizer_fn=nm,
                          weights_initializer=identity_initializer(), scope='g_conv6')
        net = slim.conv2d(net, 32, [3, 3], rate=64, activation_fn=lrelu, normalizer_fn=nm,
                          weights_initializer=identity_initializer(), scope='g_conv7')
        net = slim.conv2d(net, 32, [3, 3], rate=128, activation_fn=lrelu, normalizer_fn=nm,
                          weights_initializer=identity_initializer(), scope='g_conv8')
        net = slim.conv2d(net, 32, [3, 3], rate=1, activation_fn=lrelu, normalizer_fn=nm,
                          weights_initializer=identity_initializer(), scope='g_conv9')
        infer = slim.conv2d(net, 3, [1, 1], rate=1, activation_fn=None, scope='g_conv_last')

    return infer
