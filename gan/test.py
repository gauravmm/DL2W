# Train DCGAN model on CIFAR-10 data
# Written by Kingsley Kuan

import tensorflow as tf
from data import cifar10, utilities
from . import dcgan
import logging

logger = logging.getLogger("gan.test")

BATCH_SIZE = 256

# Sample noise from random normal distribution

random_z = tf.random_normal([BATCH_SIZE, 100], mean=0.0, stddev=1.0, name='random_z')

# Generate images with generator
generator = dcgan.generator(random_z, is_training=True, name='generator')

# Add summaries to visualise output images and losses
generator_visualisation = tf.cast(((generator / 2.0) + 0.5) * 255.0, tf.uint8)
summary_generator = tf.summary. \
                    image('summary/generator', generator_visualisation, max_outputs=8)

sv = tf.train.Supervisor(logdir="gan/train_logs/", save_summaries_secs=None, save_model_secs=None)

batch = 0
with sv.managed_session() as sess:
    logwriter = tf.summary.FileWriter("gan/test_logs/", sess.graph)
    while not sv.should_stop():
        if batch > 0 and batch % 100 == 0:
            logger.debug('Step {}.'.format(batch))
        sum_gen = sess.run(summary_generator)
        logwriter.add_summary(sum_gen, batch)
        batch += 1
