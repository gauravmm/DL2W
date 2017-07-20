"""
Simple implementation of SSD using tf.layers.

Uses only one feature map to produce detections and uses only one default box
per feature map receptive field.

Reference:
SSD: Single Shot MultiBox Detector https://arxiv.org/abs/1512.02325

Author: Kingsley Kuan
"""

import tensorflow as tf
import resnet

kernel_init = tf.contrib.layers.variance_scaling_initializer()
kernel_reg = tf.contrib.layers.l2_regularizer(0.00001)

def ssd_resnet_v2_18(inputs, num_classes, training=False, scope=None):
    with tf.variable_scope(scope, 'ssd_resnet_v2_18'):
        net = inputs
        net = resnet.resnet_v2_18(inputs, num_classes, training=training)

        # Get conv5 from ResNet-18
        ops = tf.get_default_graph().get_operations()
        ops = [op for op in ops if 'conv5' in op.name]
        net = ops[-1].outputs[0]

        # conv6_x
        net = resnet.unit(net, 512, 2, training=training, scope='conv6_1')
        net = resnet.unit(net, 512, 1, training=training, scope='conv6_2')

        with tf.variable_scope('postact'):
            net = tf.layers.batch_normalization(net, training=training,
                                                name='batch_norm')
            net = tf.nn.relu(net)

        # Output maps for object class (cls) and bounding box regression (reg)
        det_cls = tf.layers.conv2d(net, num_classes, 1, strides=1,
                                   padding='same',
                                   kernel_initializer=kernel_init,
                                   kernel_regularizer=kernel_reg,
                                   name='det_cls')

        det_reg = tf.layers.conv2d(net, 4, 1, strides=1,
                                   padding='same',
                                   kernel_initializer=kernel_init,
                                   kernel_regularizer=kernel_reg,
                                   name='det_reg')

        return det_cls, det_reg
