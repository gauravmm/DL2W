#! python3

import logging

import tensorflow as tf
from data import nodules, utilities

from . import resnet_utils_3d as utils
from . import resnet_v2_3d as resnet

slim = tf.contrib.slim


logging.basicConfig(level=logging.INFO)
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Config:
BATCH_SIZE = 32
NUM_EPOCHS = 1.5
LEARNING_RATE = 0.0001
OPTIMIZER = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
DATASET_SIZE = 13400

# Set up training data:
NUM_BATCHES = int(NUM_EPOCHS * DATASET_SIZE / BATCH_SIZE)
data_generator = utilities.infinite_generator(nodules.get_train(), BATCH_SIZE)

# Define the model:
n_input = tf.placeholder(tf.float32, shape=(None, 32,32,32,1), name="input")
n_label = tf.placeholder(tf.int64, shape=(None,), name="label")

# Build the model
with slim.arg_scope(resnet.resnet_arg_scope()):
    net, end_points = resnet.resnet_v2_18(n_input, num_classes=2, is_training=True)

# Define the loss function
loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=net, labels=n_label, name="softmax"))
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(net, axis=1), n_label), tf.float32))

reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
reg_loss = tf.add_n(reg_losses)

loss += reg_loss

# Add summaries to track the state of training:
tf.summary.scalar('summary/loss', loss)
tf.summary.scalar('summary/accuracy', accuracy)
tf.summary.histogram('labels', n_label)
tf.summary.histogram('predictions', tf.nn.softmax(net))
summaries = tf.summary.merge_all()

# Define training operations:
global_step = tf.Variable(0, trainable=False, name='global_step')
inc_global_step = tf.assign(global_step, global_step+1)

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train_op = OPTIMIZER.minimize(loss)

logger.info("Loading training supervisor...")
sv = tf.train.Supervisor(logdir="3dcnn/train_logs/", global_step=global_step, summary_op=None, save_model_secs=30)
logger.info("Done!")

with sv.managed_session() as sess:
    # Get the current global_step
    batch = sess.run(global_step)

    # Set up tensorboard logging:
    logwriter = tf.summary.FileWriter("3dcnn/train_logs/", sess.graph)
    logwriter.add_session_log(tf.SessionLog(status=tf.SessionLog.START), global_step=batch)

    logger.info("Starting training from batch {} to {}. Saving model every {}s.".format(batch, NUM_BATCHES, 30))

    while not sv.should_stop():
        if batch >= NUM_BATCHES:
            logger.info("Saving...")
            sv.saver.save(sess, "3dcnn/train_logs/model.ckpt", global_step=batch)
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
