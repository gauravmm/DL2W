from data import utilities

import logging
logger = logging.getLogger(__name__)

def get_train():
    files = ['data.tfrecord']
    return utilities.ensure_dataset_exists(files, 'simple_ssd/train')

def get_test():
    files = ['000067.png', '000076.png', '000098.png', '000103.png', '000164.png']
    return utilities.ensure_dataset_exists(files, 'simple_ssd/test')
