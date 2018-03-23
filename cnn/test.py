#! /usr/bin/env python3

import logging

import tensorflow as tf
from data import cifar10, utilities

from . import vgg

logger = logging.getLogger(__name__)

# Config:
BATCH_SIZE = 64

# Data:
data_generator = utilities.finite_generator(cifar10.get_test(), BATCH_SIZE)

# Define the model:
n_input = tf.placeholder(tf.float32, shape=cifar10.get_shape_input(), name="input")
n_label = tf.placeholder(tf.int64, shape=cifar10.get_shape_label(), name="label")

# Build the model
n_output = vgg.build(n_input)
accuracy = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(n_output, axis=1), n_label), tf.int32))
global_step = tf.Variable(0, trainable=False, name='global_step')

# Model loader
pre_train_saver = tf.train.Saver()
load_pretrain = lambda sess: pre_train_saver.restore(sess, "cnn/train_logs/")

logger.info("Loading training supervisor...")
sv = tf.train.Supervisor(logdir="cnn/train_logs/", init_fn=load_pretrain, global_step=global_step, summary_op=None, save_model_secs=None)
logger.info("Done!")

with sv.managed_session() as sess:
    batch = sess.run(global_step)
    correct = 0
    total = 0
    logger.info("Testing performance from batch {}.".format(batch))

    try:
        while not sv.should_stop():
            if batch > 0 and batch % 100 == 0:
                logger.debug('Testing step {}.'.format(batch))
            inp, lbl = next(data_generator)
            total += len(lbl)
            correct += sess.run(accuracy, feed_dict={
                n_input: inp,
                n_label: lbl
            })
    except StopIteration:
        logger.info("Done!")
        logger.info("{:d}/{:d} correct; {:.2f}%".format(correct, total, 100.0*correct/float(total)))
        pass

logger.info("Halting.")
