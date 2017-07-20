"""
Modified VGG suitable for CIFAR-10
"""

import logging

import tensorflow as tf

logger = logging.getLogger(__name__)

pad = 'SAME'

def build(x):
    assert str(x.get_shape()) == "(?, 32, 32, 3)"
    logger.info("Building VGG model")

    log = lambda x: logger.info("\t{}\t{}".format(x.get_shape(), x.name))

    log(x)
    with tf.variable_scope('conv1'):
        x = tf.layers.conv2d(x, 64, (3, 3), padding=pad, activation=tf.nn.relu, name="conv1_1")
        x = tf.layers.conv2d(x, 64, (3, 3), padding=pad, activation=tf.nn.relu, name="conv1_2")
        x = tf.layers.max_pooling2d(x, (2, 2), (2, 2), name="pool1")

    log(x)
    with tf.variable_scope('conv2'):
        x = tf.layers.conv2d(x, 128, (3, 3), padding=pad, activation=tf.nn.relu, name="conv2_1")
        x = tf.layers.conv2d(x, 128, (3, 3), padding=pad, activation=tf.nn.relu, name="conv2_2")
        x = tf.layers.max_pooling2d(x, (2, 2), (2, 2), name="pool2")

    log(x)
    with tf.variable_scope('conv3'):
        x = tf.layers.conv2d(x, 256, (3, 3), padding=pad, activation=tf.nn.relu, name="conv3_1")
        x = tf.layers.conv2d(x, 256, (3, 3), activation=tf.nn.relu, name="conv3_2")
        x = tf.layers.conv2d(x, 256, (3, 3), activation=tf.nn.relu, name="conv3_3")
        x = tf.layers.conv2d(x, 256, (3, 3), activation=tf.nn.relu, name="conv3_4")
        x = tf.layers.max_pooling2d(x, (2, 2), (2, 2), name="pool3")

    log(x)
    x = tf.layers.dense(x, 1024, activation=tf.nn.relu, name="fc4")

    log(x)
    x = tf.layers.dense(x, 256, activation=tf.nn.relu, name="fc5")

    log(x)
    x = tf.layers.dense(x, 10, activation=tf.nn.relu, name="fc6")

    # Remove singular dimensions:
    log(x)
    x = tf.squeeze(x, [1, 2])

    log(x)
    return x
