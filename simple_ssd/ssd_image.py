"""
Represents an image with bounding boxes and associated SSD groundtruth.

Author: Kingsley Kuan
"""

import math
import numpy as np
import tensorflow as tf
from PIL import Image
from bounding_box import BoundingBox

LABELS = ['Background', 'Car', 'Pedestrian', 'Cyclist']
LABEL_IDS = dict(zip(LABELS, range(len(LABELS))))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

class SSDImage(object):
    """Represents an image with bounding boxes and associated groundtruth."""
    def __init__(self, image_id, image_path, stride, classes, bboxes=None):
        self.image_id = image_id
        self.image_path = image_path
        self.stride = stride
        self.classes = classes
        self.bboxes = bboxes

        image = Image.open(self.image_path)
        self.width = float(image.width)
        self.height = float(image.height)

        # Calculate width and height of cls and reg groundtruth feature maps
        self.feature_map_width = int(math.ceil(self.width / self.stride))
        self.feature_map_height = int(math.ceil(self.height / self.stride))

        # Object class (cls) and bounding box regression (reg) groundtruth
        # feature maps
        self.cls_map = np.zeros((self.feature_map_height,
                                 self.feature_map_width,
                                 self.classes), dtype=np.float32)
        self.reg_map = np.zeros((self.feature_map_height,
                                 self.feature_map_width,
                                 4), dtype=np.float32)

        # Match groundtruth bounding boxes to default boxes
        if bboxes is not None:
            self.match_default_boxes()

    def match_default_boxes(self):
        """Match groundtruth bounding boxes to default boxes."""
        default_boxes = [[None for x in range(self.feature_map_width)]
                         for y in range(self.feature_map_height)]

        # Create default boxes with one scale and aspect ratio
        for y in range(self.feature_map_height):
            for x in range(self.feature_map_width):
                default_boxes[y][x] = BoundingBox(x * self.stride,
                                                  y * self.stride,
                                                  x * self.stride + self.stride,
                                                  y * self.stride + self.stride)

        # NumPy array to hold iou overlap between groundtruth bounding boxes
        # and all default boxes
        bbox_ious = np.zeros((self.feature_map_height,
                              self.feature_map_width,
                              len(self.bboxes)))

        # Calculate iou overlap between groundtruth bounding boxes and
        # all default boxes
        for y in range(self.feature_map_height):
            for x in range(self.feature_map_width):
                for i, bbox in enumerate(self.bboxes):
                    iou = bbox.intersection_over_union(default_boxes[y][x])
                    bbox_ious[y, x, i] = iou

        # Find the coordinates of the best matching default box for each
        # groundtruth bounding box
        best_iou_coord_per_bbox = []
        for i, bbox in enumerate(self.bboxes):
            best_iou_coord = np.unravel_index(np.argmax(bbox_ious[:, :, i]),
                                              bbox_ious[:, :, i].shape)
            best_iou_coord_per_bbox.append(best_iou_coord)

        # For each default box
        for y in range(self.feature_map_height):
            for x in range(self.feature_map_width):
                # If no bounding box matches, assign background class
                if np.max(bbox_ious[y, x]) == 0.0:
                    self.cls_map[y, x, 0] = 1
                    continue

                # Get index of the bounding box with the highest iou overlap
                best_iou_bbox = np.argmax(bbox_ious[y, x])

                # Get iou overlap of this bounding box
                iou = bbox_ious[y, x, best_iou_bbox]

                # Get coordinate of the best matching default box for this
                # bounding box
                bbox_best_iou_coord = best_iou_coord_per_bbox[best_iou_bbox]

                # Assign the ground groundtruth bounding box to this default box
                # if either iou overlap is greater than 0.5 or if it is the best
                # match for the bounding box
                if iou >= 0.5 or (bbox_best_iou_coord[0] == y and
                                  bbox_best_iou_coord[1] == x):
                    self.assign_default_box(x, y,
                                            self.bboxes[best_iou_bbox],
                                            default_boxes[y][x])
                else:
                    # Otherwise, assign background class
                    self.cls_map[y, x, 0] = 1

        return self

    def assign_default_box(self, x, y, bbox, default_bbox):
        """Encodes default boxes into the class and regression feature maps"""
        self.cls_map[y, x, bbox.label_id] = 1

        x_reg, y_reg, width_reg, height_reg = bbox.get_regression(default_bbox)
        self.reg_map[y, x, 0] = x_reg
        self.reg_map[y, x, 1] = y_reg
        self.reg_map[y, x, 2] = width_reg
        self.reg_map[y, x, 3] = height_reg

    def to_tf_example(self):
        """Converts the image and associated groundtruth into a TF Example."""
        with open(self.image_path, 'rb') as file:
            image = file.read()

        tf_example = tf.train.Example(features=tf.train.Features(feature={
            'image_id': _bytes_feature(self.image_id.encode()),
            'image': _bytes_feature(image),
            'cls_map': _bytes_feature(self.cls_map.tobytes()),
            'reg_map': _bytes_feature(self.reg_map.tobytes()),
        }))

        return tf_example

    @staticmethod
    def read_kitti_data(image_id, image_path, label_path, stride):
        """Reads groundtruth data from a Kitti annotation file."""
        with open(label_path) as file:
            data = file.read()

        bboxes = []

        lines = data.splitlines()
        for line in lines:
            line = line.split()
            if line[0] in LABEL_IDS:
                bboxes.append(
                    BoundingBox(line[4],
                                line[5],
                                line[6],
                                line[7],
                                LABEL_IDS[line[0]]))

        return SSDImage(image_id, image_path, stride, len(LABELS),
                        bboxes=bboxes)
