#! /usr/bin/env python3

"""
Generates SSD default boxes and Kitti data as TFRecords.

Author: Kingsley Kuan
"""

import argparse
import os
import tensorflow as tf
from ssd_image import SSDImage

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--image_dir', type=str, default='training/image_2',
                        help='Directory where Kitti images are located.')

    parser.add_argument('--label_dir', type=str, default='training/label_2',
                        help='Directory where Kitti labels are located.')

    parser.add_argument('--tfrecord_file', type=str, default='data.tfrecord',
                        help='TFRecord file to write to.')

    parser.add_argument('--stride', type=int, default=64,
                        help='Final stride of the detection framework.')

    return parser.parse_args()

def box_generator(image_dir, label_dir, tfrecord_file, stride):
    """Generates SSD default boxes and saves Kitti data as TFRecords."""
    # Read in all Kitti image metadata and generates SSD default boxes
    kitti_images = []
    for filename in os.listdir(label_dir):
        image_id = os.path.splitext(filename)[0]
        image_path = os.path.join(image_dir, '{}.png'.format(image_id))
        label_path = os.path.join(label_dir, '{}.txt'.format(image_id))
        kitti_images.append(
            SSDImage.read_kitti_data(image_id, image_path, label_path, stride))

    # Convert all images into TF Examples and write to a TFRecord.
    with tf.python_io.TFRecordWriter(tfrecord_file) as writer:
        for kitti_image in kitti_images:
            tf_example = kitti_image.to_tf_example()
            writer.write(tf_example.SerializeToString())

if __name__ == '__main__':
    args = parse_args()
    box_generator(args.image_dir, args.label_dir,
                  args.tfrecord_file, args.stride)
