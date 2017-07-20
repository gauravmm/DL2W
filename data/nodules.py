import logging

import numpy as np

from . import utilities

logger = logging.getLogger(__name__)

def get_train():
    fn = _get_files("train")
    logger.info("Loading training data from {}".format(fn))
    return _load_file(fn)

def get_test():
    fn = _get_files("test")
    logger.info("Loading test data from {}".format(fn))
    return _load_file(fn)

def get_shape_input():
    return (None, 32, 32, 32, 1)

def get_shape_label():
    return (None,)

def num_classes():
    return 2

def _get_files(split):
    files=["test_nodules_and_labels.npz", "training_nodules_and_labels.npz"]
    fn = [v for v in utilities.ensure_dataset_exists(files, "nodules") if split in v].pop()
    return fn

def _load_file(fn):
    with np.load(fn) as data:
        return ((np.expand_dims(data["data"], axis=-1) - (-5.47))/(24.2 -(-5.47)), data["labels"])
