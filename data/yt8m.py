import logging

import numpy as np

from data import utilities

logger = logging.getLogger(__name__)

def get_train():
    files=["testgz.tfrecord", "testIa.tfrecord", "testYI.tfrecord", "testzd.tfrecord", "traindw.tfrecord", "traineA.tfrecord", "trainha.tfrecord", "trainI1.tfrecord", "trainjw.tfrecord"]
    return utilities.ensure_dataset_exists(files, "yt8m")

def get_test():
    files=["testgz.tfrecord", "testIa.tfrecord", "testYI.tfrecord", "testzd.tfrecord", "traindw.tfrecord", "traineA.tfrecord", "trainha.tfrecord", "trainI1.tfrecord", "trainjw.tfrecord"]
    return utilities.ensure_dataset_exists(files, "yt8m")
