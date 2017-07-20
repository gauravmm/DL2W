#!/usr/bin/env python3

"""
Predicts object locations on a single image.

Author: Kingsley Kuan
"""

import os 
import sys 
sys.path.append(os.path.dirname(__file__))

import argparse
import os
import math
import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw
import ssd
import bounding_box

IMAGE_WIDTH = 1242
IMAGE_HEIGHT = 375
STRIDE = 64
MAP_WIDTH = int(math.ceil(IMAGE_WIDTH / STRIDE))
MAP_HEIGHT = int(math.ceil(IMAGE_HEIGHT / STRIDE))
NUM_CLASSES = 4

SCORE_THRESHOLD = 0.3
NMS_THRESHOLD = 0.5

CLASS_COLOUR_DICT = {1: 'red', 2: 'green', 3: 'blue'}

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_image_path', type=str, default='test.png',
                        help='Path of image to detect objects in.')

    parser.add_argument('--output_image_path', type=str, default='output.png',
                        help='Path to write output image.')

    parser.add_argument('--checkpoint_dir', type=str, default='train_logs',
                        help='Directory to read model checkpoint.')

    return parser.parse_args()

def test(input_image_path, output_image_path, checkpoint_dir):
    """Predicts object locations on a single image."""

    # Placeholder to pass raw image into
    raw_image = tf.placeholder(tf.string)

    # Decode image, resize, and standardize (as in training)
    image = tf.image.decode_png(raw_image, 3)
    image = tf.image.resize_images(image, (IMAGE_HEIGHT, IMAGE_WIDTH))
    image = tf.image.per_image_standardization(image)
    image = tf.expand_dims(image, axis=0)

    # Detect object classes and bounding box regression for each default box
    det_cls, det_reg = ssd.ssd_resnet_v2_18(image, NUM_CLASSES, training=False)
    det_cls = tf.nn.softmax(det_cls)

    # Create saver to restore model from checkpoint
    saver = tf.train.Saver()

    with tf.Session() as sess, open(input_image_path, 'rb') as file:
        # Restore model from latest checkpoint
        saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir))

        # Run object detection
        det_cls_out, det_reg_out = \
            sess.run((det_cls, det_reg), feed_dict={raw_image: file.read()})

    # Slice out object scores and bounding box regression
    det_cls_out = det_cls_out[0, :, :, 1:]
    det_reg_out = det_reg_out[0]
    top_classes = np.argmax(det_cls_out, axis=-1)

    bboxes = []
    for y in range(MAP_HEIGHT):
        for x in range(MAP_WIDTH):
            label_id = top_classes[y, x]

            # Skip if score is below threshold
            if det_cls_out[y, x, label_id] < SCORE_THRESHOLD:
                continue

            # Create default box at feature map location
            default_box = bounding_box.BoundingBox(x * STRIDE,
                                                   y * STRIDE,
                                                   x * STRIDE + STRIDE,
                                                   y * STRIDE + STRIDE,
                                                   label_id + 1)

            # Apply regression to default box
            default_box.apply_regression(det_reg_out[y, x, 0],
                                         det_reg_out[y, x, 1],
                                         det_reg_out[y, x, 2],
                                         det_reg_out[y, x, 3])

            # Add bounding box to list of detected boxes along with score
            bboxes.append((default_box, det_cls_out[y, x, label_id]))

    # Sort bounding boxes by score
    bboxes.sort(key=lambda x: x[1], reverse=True)
    bboxes, _ = zip(*bboxes)

    # Use non maximum suppression to supress bounding boxes with high overlap
    suppress = []
    for i in range(len(bboxes)):
        for j in range(i+1, len(bboxes)):
            if bboxes[i].intersection_over_union(bboxes[j]) >= NMS_THRESHOLD:
                suppress.append(j)
    filtered_bboxes = []
    for i, bbox in enumerate(bboxes):
        if i not in suppress:
            filtered_bboxes.append(bbox)

    # Open image and prepare to draw on it
    im = Image.open(input_image_path)
    im_draw = ImageDraw.Draw(im)

    # Draw each bounding box on the image
    for bbox in filtered_bboxes:
        im_draw.rectangle((bbox.x_min,
                           bbox.y_min,
                           bbox.x_max,
                           bbox.y_max),
                          outline=CLASS_COLOUR_DICT[bbox.label_id])
    im.save(output_image_path)

if __name__ == '__main__':
    args = parse_args()
    test(args.input_image_path, args.output_image_path, args.checkpoint_dir)
else:
    files = \
        ['000067.png', '000076.png', '000098.png', '000103.png', '000164.png']
    for file in files:
        if not os.path.exists('simple_ssd/output'):
            os.makedirs('simple_ssd/output')

        with tf.Graph().as_default():
            test(os.path.join('data/simple_ssd/test/', file),
                 os.path.join('simple_ssd/output/', file),
                 'simple_ssd/train_logs')
