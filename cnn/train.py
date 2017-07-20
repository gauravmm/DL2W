#! python3

import logging

import tensorflow as tf
from data import cifar10, utilities

from . import vgg

logger = logging.getLogger(__name__)

# Config:
BATCH_SIZE = 64
NUM_EPOCHS = 30
LEARNING_RATE = 0.0001
OPTIMIZER = tf.train.AdamOptimizer(learning_rate = LEARNING_RATE)
DATASET_SIZE = 50000

# Set up training data:
NUM_BATCHES = int(NUM_EPOCHS * DATASET_SIZE / BATCH_SIZE)
data_generator = utilities.infinite_generator(cifar10.get_train(), BATCH_SIZE)

# Define the placeholders:
n_input = tf.placeholder(tf.float32, shape=cifar10.get_shape_input(), name="input")
n_label = tf.placeholder(tf.int64, shape=cifar10.get_shape_label(), name="label")

# Build the model
n_output = vgg.build(n_input)

# Define the loss function
loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=n_output, labels=n_label, name="softmax"))
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(n_output, axis=1), n_label), tf.float32))

# Add summaries to track the state of training:
tf.summary.scalar('summary/loss', loss)
tf.summary.scalar('summary/accuracy', accuracy)
summaries = tf.summary.merge_all()

# Define training operations:
global_step = tf.Variable(0, trainable=False, name='global_step')
inc_global_step = tf.assign(global_step, global_step+1)

train_op = OPTIMIZER.minimize(loss)

logger.info("Loading training supervisor...")
sv = tf.train.Supervisor(logdir="cnn/train_logs/", global_step=global_step, summary_op=None, save_model_secs=30)
logger.info("Done!")

with sv.managed_session() as sess:
    # Get the current global_step
    batch = sess.run(global_step)

    # Set up tensorboard logging:
    logwriter = tf.summary.FileWriter("cnn/train_logs/", sess.graph)
    logwriter.add_session_log(tf.SessionLog(status=tf.SessionLog.START), global_step=batch)

    logger.info("Starting training from batch {} to {}. Saving model every {}s.".format(batch, NUM_BATCHES, 30))

    while not sv.should_stop():
        if batch >= NUM_BATCHES:
            logger.info("Saving...")
            sv.saver.save(sess, "cnn/train_logs/model.ckpt", global_step=batch)
            sv.stop()
            break

        if batch > 0 and batch % 100 == 0:
            logger.info('Step {} of {}.'.format(batch, NUM_BATCHES))

        inp, lbl = next(data_generator)

        summ, _ = sess.run((summaries, (train_op, inc_global_step)), feed_dict={
            n_input: inp,
            n_label: lbl
        })
        batch += 1
        
        logwriter.add_summary(summ, global_step=batch)

logger.info("Halting.")
