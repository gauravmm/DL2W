"""
Represents a bounding box for use in object detection.

Author: Kingsley Kuan
"""

import math

class BoundingBox(object):
    """Represents a bounding box for use in object detection."""
    def __init__(self, x_min, y_min, x_max, y_max, label_id=None):
        self.x_min = float(x_min)
        self.y_min = float(y_min)
        self.x_max = float(x_max)
        self.y_max = float(y_max)
        self.label_id = label_id

        self.x_center = (self.x_min + self.x_max) / 2.0
        self.y_center = (self.y_min + self.y_max) / 2.0
        self.width = self.x_max - self.x_min
        self.height = self.y_max - self.y_min

    def area(self):
        return (self.x_max - self.x_min) * (self.y_max - self.y_min)

    def intersection_over_union(self, bbox):
        """Computes the iou or jaccard overlap between two bounding boxes."""
        intersection_box = BoundingBox(max(self.x_min, bbox.x_min),
                                       max(self.y_min, bbox.y_min),
                                       min(self.x_max, bbox.x_max),
                                       min(self.y_max, bbox.y_max))

        # Return 0 if the bounding boxes have no overlap
        if intersection_box.x_min >= intersection_box.x_max or \
           intersection_box.y_min >= intersection_box.y_max:
            return 0.0

        intersection_area = intersection_box.area()

        union_area = self.area() + bbox.area() - intersection_area
        return intersection_area / union_area

    def get_regression(self, bbox):
        """Return offsets between two bounding boxes."""
        x_reg = (self.x_center - bbox.x_center) / bbox.width
        y_reg = (self.y_center - bbox.y_center) / bbox.height
        width_reg = math.log(self.width / bbox.width)
        height_reg = math.log(self.height / bbox.height)
        return x_reg, y_reg, width_reg, height_reg

    def apply_regression(self, x_reg, y_reg, width_reg, height_reg):
        """Apply offsets to bounding box."""
        self.x_center += x_reg * self.width
        self.y_center += y_reg * self.height
        self.width *= math.exp(width_reg)
        self.height *= math.exp(height_reg)

        self.x_min = self.x_center - (self.width / 2.0)
        self.y_min = self.y_center - (self.height / 2.0)
        self.x_max = self.x_center + (self.width / 2.0)
        self.y_max = self.y_center + (self.height / 2.0)

        return self

    def __str__(self):
        return '{}, {}, {}, {}'.format(self.x_min,
                                       self.y_min,
                                       self.x_max,
                                       self.y_max)
