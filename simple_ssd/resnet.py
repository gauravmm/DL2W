"""
Implementation of pre-activation version of ResNet-18 using tf.layers.

Reference:
Deep Residual Learning for Image Recognition https://arxiv.org/abs/1512.03385
Identity Mappings in Deep Residual Networks https://arxiv.org/abs/1603.05027

Author: Kingsley Kuan
"""

import tensorflow as tf

kernel_init = tf.contrib.layers.variance_scaling_initializer()
kernel_reg = tf.contrib.layers.l2_regularizer(0.00001)

def unit(inputs, channels, stride, training=False, scope=None):
    with tf.variable_scope(scope, 'conv'):
        with tf.variable_scope('preact'):
            preact = tf.layers.batch_normalization(inputs, training=training,
                                                   name='batch_norm')
            preact = tf.nn.relu(preact)

        channels_in = inputs.shape.dims
        if channels == channels_in and stride == 1:
            shortcut = tf.identity(inputs, name='shortcut')
        else:
            shortcut = tf.layers.conv2d(preact, channels, 1, strides=stride,
                                        padding='same',
                                        kernel_initializer=kernel_init,
                                        kernel_regularizer=kernel_reg,
                                        name='shortcut')

        with tf.variable_scope('residual'):
            residual = preact

            residual = tf.layers.conv2d(residual, channels, 3, strides=stride,
                                        padding='same',
                                        kernel_initializer=kernel_init,
                                        kernel_regularizer=kernel_reg,
                                        name='conv1')
            residual = tf.layers.batch_normalization(residual,
                                                     training=training,
                                                     name='conv1/batch_norm')
            residual = tf.nn.relu(residual, name='conv1/relu')

            residual = tf.layers.conv2d(residual, channels, 3, strides=1,
                                        padding='same',
                                        kernel_initializer=kernel_init,
                                        kernel_regularizer=kernel_reg,
                                        name='conv2')

        return shortcut + residual

def resnet_v2_18(inputs, num_classes, training=False, scope=None):
    with tf.variable_scope(scope, 'resnet_v2_18'):
        net = inputs

        # conv1
        net = tf.layers.conv2d(net, 64, 7, strides=2, padding='same',
                               kernel_initializer=kernel_init,
                               kernel_regularizer=kernel_reg,
                               name='conv1')
        net = tf.layers.max_pooling2d(net, 3, strides=2, padding='same',
                                      name='max_pool')

        # conv2_x
        net = unit(net, 64, 1, training=training, scope='conv2_1')
        net = unit(net, 64, 1, training=training, scope='conv2_2')

        # conv3_x
        net = unit(net, 128, 2, training=training, scope='conv3_1')
        net = unit(net, 128, 1, training=training, scope='conv3_2')

        # conv4_x
        net = unit(net, 256, 2, training=training, scope='conv4_1')
        net = unit(net, 256, 1, training=training, scope='conv4_2')

        # conv5_x
        net = unit(net, 512, 2, training=training, scope='conv5_1')
        net = unit(net, 512, 1, training=training, scope='conv5_2')

        # Post activation at the end of the resnet blocks
        with tf.variable_scope('postact'):
            net = tf.layers.batch_normalization(net, training=training,
                                                name='batch_norm')
            net = tf.nn.relu(net)

        # Global average pooling
        net = tf.reduce_mean(net, (1, 2), name='global_avg_pool')

        # Fully connected layer outputs class probabilities
        net = tf.layers.dense(net, num_classes,
                              kernel_initializer=kernel_init,
                              kernel_regularizer=kernel_reg,
                              name='output')

        return net
