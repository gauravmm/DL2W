"""
Creates TensorFlow input pipeline for Kitti SSD data.

Author: Kingsley Kuan
"""

import math
import tensorflow as tf

IMAGE_WIDTH = 1242
IMAGE_HEIGHT = 375
STRIDE = 64
MAP_WIDTH = int(math.ceil(IMAGE_WIDTH / STRIDE))
MAP_HEIGHT = int(math.ceil(IMAGE_HEIGHT / STRIDE))
NUM_CLASSES = 4

def decode(serialized_example):
    """Decodes single TF Example."""

    # Parse feature keys in TFRecords example
    features = tf.parse_single_example(serialized_example, features={
        'image_id': tf.FixedLenFeature([], tf.string),
        'image': tf.FixedLenFeature([], tf.string),
        'cls_map': tf.FixedLenFeature([], tf.string),
        'reg_map': tf.FixedLenFeature([], tf.string)
    })

    image_id = features['image_id']
    image = features['image']
    cls_map = tf.decode_raw(features['cls_map'], tf.float32)
    reg_map = tf.decode_raw(features['reg_map'], tf.float32)

    # Decode png image and resize to fixed size
    image = tf.image.decode_png(image, 3)
    image = tf.image.resize_images(image, (IMAGE_HEIGHT, IMAGE_WIDTH))

    # Random augmentations for brightness, saturation, hue, and contrast
    image = tf.image.random_brightness(image, max_delta=32. / 255.)
    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    image = tf.image.random_hue(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.5, upper=1.5)

    # Standardize image to have zero mean and unit norm
    image = tf.image.per_image_standardization(image)

    # Reshape object class (cls) and bounding box regression (reg) maps
    cls_map = tf.reshape(cls_map, (MAP_HEIGHT, MAP_WIDTH, NUM_CLASSES))
    reg_map = tf.reshape(reg_map, (MAP_HEIGHT, MAP_WIDTH, 4))

    return image_id, image, cls_map, reg_map

def inputs(filename, batch_size=4, num_threads=8, capacity=128,
           min_after_dequeue=16, num_epochs=None, training=True):
    with tf.name_scope('input'):
        # Add TFRecords file to queue
        filename_queue = tf.train.string_input_producer(
            [filename],
            shuffle=training,
            num_epochs=num_epochs)

        # Create TFRecords reader
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)

        # Decode TFRecords example
        features = decode(serialized_example)

        # Create operation to batch multiple TFRecords examples
        # Shuffle batches if training
        if training:
            features_batch = tf.train.shuffle_batch(
                features,
                batch_size=batch_size, num_threads=num_threads,
                capacity=capacity, min_after_dequeue=min_after_dequeue,
                allow_smaller_final_batch=True)
        else:
            features_batch = tf.train.batch(
                features,
                batch_size=batch_size, num_threads=num_threads,
                capacity=capacity, allow_smaller_final_batch=True)

        return features_batch
