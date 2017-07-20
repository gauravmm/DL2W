from data import utilities

import logging
logger = logging.getLogger(__name__)

def get_files():
    files=["{:0>2}.jpg".format(i) for i in range(1,21)]
    return utilities.ensure_dataset_exists(files, "test")

def get_train():
    rv = get_files()
    logger.info(rv)
    return [v for v in rv if "/0" in v]

def get_test():
    rv = get_files()
    logger.info(rv)
    return [v for v in rv if "/0" not in v]
