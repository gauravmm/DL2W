from data import utilities

import logging
logger = logging.getLogger(__name__)

def get_train():
    files = ['data.tfrecord', 'video_titles.json', 'video_titles_raw.json']
    other_files = ['label_names.csv', 'labels.json']
    utilities.ensure_dataset_exists(other_files, 'textcnn')
    return utilities.ensure_dataset_exists(files, 'textcnn/data_train')

def get_test():
    files = ['data.tfrecord', 'video_titles.json', 'video_titles_raw.json']
    other_files = ['label_names.csv', 'labels.json']
    utilities.ensure_dataset_exists(other_files, 'textcnn')
    return utilities.ensure_dataset_exists(files, 'textcnn/data_val')
