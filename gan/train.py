# Train DCGAN model on CIFAR-10 data
# Written by Kingsley Kuan

import tensorflow as tf
from data import cifar10, utilities
from . import dcgan
import logging

logger = logging.getLogger("gan.train")

# Create CIFAR-10 input
BATCH_SIZE = 32
data_generator = map((lambda inp: (inp[0]*2. - 1., inp[1])), utilities.infinite_generator(cifar10.get_train(), BATCH_SIZE))

# Input images
input_placeholder = tf.placeholder(tf.float32, shape=cifar10.get_shape_input(), name="input")

# Sample noise from random normal distribution
random_z = tf.random_normal([BATCH_SIZE, 100], mean=0.0, stddev=1.0,
                            name='random_z')

# Generate images with generator
generator = dcgan.generator(random_z, is_training=True, name='generator')

# Pass real and fake images into discriminator separately
real_discriminator = dcgan.discriminator(input_placeholder, is_training=True,
                                         name='discriminator')
fake_discriminator = dcgan.discriminator(generator, is_training=True,
                                         reuse=True, name='discriminator')

# Calculate seperate losses for discriminator with real and fake images
real_discriminator_loss = tf.losses.sigmoid_cross_entropy(
    tf.constant(1, shape=[BATCH_SIZE]),
    real_discriminator,
    scope='real_discriminator_loss')
fake_discriminator_loss = tf.losses.sigmoid_cross_entropy(
    tf.constant(0, shape=[BATCH_SIZE]),
    fake_discriminator,
    scope='fake_discriminator_loss')

# Add discriminator losses
discriminator_loss = real_discriminator_loss + fake_discriminator_loss

# Calculate loss for generator by flipping label on discriminator output
generator_loss = tf.losses.sigmoid_cross_entropy(
    tf.constant(1, shape=[BATCH_SIZE]),
    fake_discriminator,
    scope='generator_loss')


# Add summaries to visualise output images and losses
summary_discriminator = tf.summary.merge([ \
    tf.summary.scalar('summary/real_discriminator_loss', real_discriminator_loss), \
    tf.summary.scalar('summary/fake_discriminator_loss', fake_discriminator_loss), \
    tf.summary.scalar('summary/discriminator_loss', discriminator_loss)])

generator_visualisation = tf.cast(((generator / 2.0) + 0.5) * 255.0, tf.uint8)
summary_generator = tf.summary.merge([ \
tf.summary.image('summary/generator', generator_visualisation, max_outputs=32), \
tf.summary.scalar('summary/generator_loss', generator_loss)])

# Get discriminator and generator variables to train separately
discriminator_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                            scope='discriminator')
generator_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                            scope='generator')

# Get discriminator and generator update ops
discriminator_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS,
                                             scope='discriminator')
generator_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS,
                                         scope='generator')

# Train discriminator first followed by generator
global_step = tf.Variable(0, name='global_step', trainable=False)

with tf.control_dependencies(discriminator_update_ops):
    train_discriminator = tf.train. \
                          AdamOptimizer(learning_rate=0.0002, beta1=0.5). \
                          minimize(discriminator_loss,
                                   var_list=discriminator_variables)

with tf.control_dependencies(generator_update_ops):
    train_generator = tf.train. \
                      AdamOptimizer(learning_rate=0.0002, beta1=0.5). \
                      minimize(generator_loss, global_step=global_step,
                               var_list=generator_variables)


# We disable automatic summaries here, because the automatic system assumes that
# any time you run any part of the graph, you will be providing values for _all_
# summaries:
sv = tf.train.Supervisor(logdir="gan/train_logs/", global_step=global_step,
                         save_summaries_secs=None, save_model_secs=120)

batch = 0
with sv.managed_session() as sess:
    # Set up tensorboard logging:
    logwriter = tf.summary.FileWriter("gan/train_logs/", sess.graph)

    while not sv.should_stop():
        if batch > 0 and batch % 100 == 0:
            logger.debug('Step {}.'.format(batch))

        inp, _ = next(data_generator)
        (_, sum_dis) = sess.run((train_discriminator, summary_discriminator), feed_dict={input_placeholder: inp})
        logwriter.add_summary(sum_dis)
        (_, sum_gen) = sess.run((train_generator, summary_generator))
        logwriter.add_summary(sum_gen)

        batch += 1
