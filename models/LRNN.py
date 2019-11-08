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


def compute_output_shape(self, input_shape):
    return tuple(input_shape[0])


def reorder_input( horizontal,reverse,X):
    # X.shape = (batch_size, row, column, channel)
    if horizontal:
        X = tf.transpose(X, (2, 0, 1, 3))
    else:
        X = tf.transpose(X, (1, 0, 2, 3))
    if reverse:
        X = tf.reverse(X, [0])
    return X


def reorder_output( horizontal,reverse,X):
    if reverse:
        X = tf.reverse(X, [0])
    if horizontal:
        X = tf.transpose(X, (1, 2, 0, 3))
    else:
        X = tf.transpose(X, (1, 0, 2, 3))
    return X


def LRNN(horizontal, reverse, X,G):
    X = reorder_input(horizontal,reverse,X)
    G = reorder_input(horizontal,reverse,G)

    def compute(a, x):
        H = a
        X, G = x
        L = H - X
        H = G * L + X
        return H
    initializer = tf.zeros_like(X[0])
    S = tf.scan(compute, (X, G), initializer)
    H = reorder_output(horizontal,reverse,S)
    return H


def net(x, reuse):
    with tf.variable_scope('LRNN',reuse=reuse):
        scale0 = slim.conv2d(x, 3, [3, 3], rate=1, activation_fn=None, normalizer_fn=None, scope='scale0')

        scale1=slim.max_pool2d(scale0,kernel_size=2)
        scale2=slim.max_pool2d(scale1,kernel_size=2)
        scale3=slim.max_pool2d(scale2,kernel_size=2)
        resize1=tf.image.resize_bilinear(scale1,tf.shape(x)[1:3])
        resize2=tf.image.resize_bilinear(scale2,tf.shape(x)[1:3])
        resize3=tf.image.resize_bilinear(scale3,tf.shape(x)[1:3])
        concat0=tf.concat([scale0,resize1,resize2,resize3],axis=3)

        conv0 = slim.conv2d(concat0, 16, [3, 3], rate=1, activation_fn=None, normalizer_fn=None, scope='conv0')

        conv1 = slim.conv2d(x, 16, [5, 5], rate=1, activation_fn=tf.nn.relu, normalizer_fn=None, scope='conv1')
        pool1=slim.max_pool2d(conv1,kernel_size=2)
        conv2 = slim.conv2d(pool1, 32, [3, 3], rate=1, activation_fn=tf.nn.relu, normalizer_fn=None, scope='conv2')
        pool2=slim.max_pool2d(conv2,kernel_size=2)
        conv3 = slim.conv2d(pool2, 32, [3, 3], rate=1, activation_fn=tf.nn.relu, normalizer_fn=None, scope='conv3')
        pool3=slim.max_pool2d(conv3,kernel_size=2)
        conv4 = slim.conv2d(pool3, 32, [3, 3], rate=1, activation_fn=tf.nn.relu, normalizer_fn=None, scope='conv4')
        pool4=slim.max_pool2d(conv4,kernel_size=2)
        conv5 = slim.conv2d(pool4, 64, [3, 3], rate=1, activation_fn=tf.nn.relu, normalizer_fn=None, scope='conv5')
        upsample5 = tf.image.resize_bilinear(conv5, tf.shape(conv4)[1:3])
        concat5=tf.concat([conv4,upsample5],axis=3)
        conv6= slim.conv2d(concat5, 32, [3, 3], rate=1, activation_fn=tf.nn.relu, normalizer_fn=None, scope='conv6')
        upsample6 = tf.image.resize_bilinear(conv6, tf.shape(conv3)[1:3])
        concat6=tf.concat([conv3,upsample6],axis=3)
        conv7= slim.conv2d(concat6, 32, [3, 3], rate=1, activation_fn=tf.nn.relu, normalizer_fn=None, scope='conv7')
        upsample7 = tf.image.resize_bilinear(conv7, tf.shape(conv2)[1:3])
        concat7=tf.concat([conv2,upsample7],axis=3)
        conv8= slim.conv2d(concat7, 32, [3, 3], rate=1, activation_fn=tf.nn.relu, normalizer_fn=None, scope='conv8')
        upsample8 = tf.image.resize_bilinear(conv8, tf.shape(conv1)[1:3])
        concat8=tf.concat([conv1,upsample8],axis=3)
        conv9 = slim.conv2d(concat8, 64, [3, 3], rate=1, activation_fn=tf.nn.tanh, normalizer_fn=None,
                            scope='conv9')
        wx1 = conv9[:, :, :, 0:16]
        wx2 = conv9[:, :, :, 16:32]
        wy1 = conv9[:, :, :, 32:48]
        wy2 = conv9[:, :, :, 48:64]
        y1 = LRNN(horizontal=True, reverse=False,X=conv0, G=wx1)
        y2 = LRNN(horizontal=True, reverse=True,X=conv0, G=wx1)
        y3 = LRNN(horizontal=False, reverse=False,X=conv0, G=wy1)
        y4 = LRNN(horizontal=False, reverse=True,X=conv0, G=wy1)
        y5 = LRNN(horizontal=True, reverse=False,X=y1, G=wx2)
        y6 = LRNN(horizontal=True, reverse=True,X=y2, G=wx2)
        y7 = LRNN(horizontal=False, reverse=False,X=y3, G=wy2)
        y8 = LRNN(horizontal=False, reverse=True,X=y4, G=wy2)
        y_max1 = tf.maximum(y5,y6)
        y_max2 = tf.maximum(y7,y8)
        y_max = tf.maximum(y_max1,y_max2)
        y = slim.conv2d(y_max, 64, [3, 3], rate=1, activation_fn=tf.nn.relu, normalizer_fn=None, scope='y')
        infer = slim.conv2d(y, 3, [3, 3], rate=1, activation_fn=tf.nn.sigmoid, normalizer_fn=None,  scope='infer')
    return infer
