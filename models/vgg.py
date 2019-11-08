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


def block(x, name):
    b_c1 = slim.conv2d(x, 64, 3, activation_fn=tf.nn.relu, weights_initializer=init(), normalizer_fn=slim.batch_norm, scope=name+'_c1')
    b_c2 = slim.conv2d(b_c1, 64, 3, activation_fn=None, weights_initializer=init(), normalizer_fn=slim.batch_norm, scope=name+'_c2')
    return b_c2 + x


def net(x, reuse):
    with tf.variable_scope('vgg', reuse=reuse):
        # First convolution
        c1 = slim.conv2d(x, 64, 3, activation_fn=tf.nn.relu, weights_initializer=init(), biases_initializer=None, scope='c1')

        # Encoder - decoder
        b1 = block(c1, 'b1')
        b2 = block(b1, 'b2')
        b3 = block(b2, 'b3')
        b4 = block(b3, 'b4')
        b5 = block(b4, 'b5')

        # Final stage
        f1 = slim.conv2d(b5 + c1, 64, 3, activation_fn=tf.nn.relu, weights_initializer=init(), biases_initializer=None, scope='f1')
        f2 = slim.conv2d(f1, 16, 3, activation_fn=tf.nn.relu, weights_initializer=init(), biases_initializer=None, scope='f2')
        f3 = slim.conv2d(f2, 3, 3, activation_fn=None, weights_initializer=init(), biases_initializer=None, scope='f3')

    return f3
