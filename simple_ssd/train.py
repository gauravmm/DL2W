#!/usr/bin/env python3

"""
Trains SSD object detector on Kitti data.

Author: Kingsley Kuan
"""

import os 
import sys 
sys.path.append(os.path.dirname(__file__))

import argparse
import tensorflow as tf
import data_reader
import ssd

NUM_CLASSES = 4

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--tfrecord_file', type=str,
                        default='data.tfrecord',
                        help='TFRecord file to read data from.')

    parser.add_argument('--train_dir', type=str, default='train_logs',
                        help='Directory to write checkpoints and summaries.')

    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size of input data.')

    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of epochs to train for.')

    return parser.parse_args()

def train(tfrecord_file, train_dir, batch_size, num_epochs):
    """Trains SSD object detector on Kitti data."""

    # Get images, object class (cls), as well as bounding box regression (reg)
    # groundtruths from input pipeline
    with tf.device('/cpu:0'):
        image_ids, images, gt_cls, gt_reg = \
            data_reader.inputs(tfrecord_file, batch_size=batch_size,
                               num_threads=batch_size, capacity=batch_size*4,
                               min_after_dequeue=batch_size*2,
                               num_epochs=num_epochs, training=True)

    # Detect object classes and bounding box regression for each default box
    det_cls, det_reg = ssd.ssd_resnet_v2_18(images, NUM_CLASSES, training=True)

    # Slice out the groundtruth background and foreground for each default box
    gt_bg = tf.slice(gt_cls, (0, 0, 0, 0), (-1, -1, -1, 1))
    gt_fg = 1 - gt_bg

    # Cross entropy loss for object classes
    cls_loss = tf.nn.softmax_cross_entropy_with_logits(labels=gt_cls,
                                                       logits=det_cls)
    cls_loss = tf.reduce_mean(cls_loss)

    # Smooth L1 loss for bounding box regression
    # (only applied when default box is matched to an object bounding box)
    difference = gt_reg - det_reg
    reg_loss = tf.where(tf.less_equal(tf.abs(difference), 1),
                        0.5 * tf.square(difference),
                        tf.abs(difference) - 0.5)
    reg_loss = tf.reduce_mean(gt_fg * reg_loss)

    # Regularization losses for weight decay
    regularization_loss = tf.add_n(
        tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

    # Sum all losses
    total_loss = cls_loss + reg_loss + regularization_loss

    global_step = tf.Variable(0, name='global_step', trainable=False)

    # Use Adam optimiser to train network
    # Requires update ops as dependencies in order to update batch norm
    # statistics
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = tf.train.AdamOptimizer(learning_rate=0.001). \
            minimize(total_loss, global_step=global_step)

    # Calculate training accuracy
    accuracy, accuracy_op = tf.metrics.accuracy(
        tf.argmax(gt_cls, axis=-1), tf.argmax(tf.nn.softmax(det_cls), axis=-1))

    # Add summaries for Tensorboard
    tf.summary.scalar('cls_loss', cls_loss)
    tf.summary.scalar('reg_loss', reg_loss)
    tf.summary.scalar('regularization_loss', regularization_loss)
    tf.summary.scalar('accuracy', accuracy)
    tf.summary.histogram('gt_cls', tf.argmax(gt_cls, axis=-1))
    tf.summary.histogram('det_cls', tf.argmax(tf.nn.softmax(det_cls), axis=-1))
    tf.summary.histogram('det_cls_hist', tf.nn.softmax(det_cls))

    # Create supervisor to aid with setting up training and saving checkpoints
    # and summaries
    sv = tf.train.Supervisor(logdir=train_dir,
                             global_step=global_step,
                             save_summaries_secs=60,
                             save_model_secs=600)

    with sv.managed_session() as sess:
        while not sv.should_stop():
            # Run training op as well as training statistics
            _, step_out, cls_loss_out, reg_loss_out, accuracy_out, _ = \
                sess.run((train_op, global_step, cls_loss, reg_loss,
                          accuracy, accuracy_op))
            if step_out % 10 == 0:
                print("Step: {}, Cls Loss: {}, Reg Loss: {}, Accuracy: {}". \
                    format(step_out, cls_loss_out, reg_loss_out, accuracy_out))

if __name__ == '__main__':
    args = parse_args()
    train(args.tfrecord_file, args.train_dir, args.batch_size, args.num_epochs)
else:
    train('data/simple_ssd/train/data.tfrecord',
          'simple_ssd/train_logs', 16, 100)
