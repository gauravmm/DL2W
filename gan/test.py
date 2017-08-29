# Train DCGAN model on CIFAR-10 data
# Written by Kingsley Kuan

import tensorflow as tf
from data import cifar10, utilities
from . import dcgan

# Create CIFAR-10 input
BATCH_SIZE = 256
data_generator = map((lambda image, label: (image*2. - 1., label)), utilities.infinite_generator(cifar10.get_train(), BATCH_SIZE))

# Sample noise from random normal distribution
n_input = tf.placeholder(tf.float32, shape=cifar10.get_shape_input(), name="input")

random_z = tf.random_normal([BATCH_SIZE, 100], mean=0.0, stddev=1.0, name='random_z')

# Generate images with generator
generator = dcgan.generator(random_z, is_training=True, name='generator')

# Add summaries to visualise output images and losses
generator_visualisation = tf.cast(((generator / 2.0) + 0.5) * 255.0, tf.uint8)
tf.summary.image('summary/generator', generator_visualisation, max_outputs=8)

sv = tf.train.Supervisor(logdir="gan/train_logs/", save_summaries_secs=None, save_model_secs=None)

batch = 0
with sv.managed_session() as sess:
    while not sv.should_stop():
        if batch > 0 and batch % 100 == 0:
            logger.debug('Step {} of {}.'.format(batch, NUM_BATCHES))

        inp, _ = next(data_generator)
        sess.run(generator_visualisation)
        
        batch += 1
