#! /usr/bin/env python3

# Author: Kingsley Kuan

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'google_yt8m'))

import csv
import json
import numpy as np
import eval_util

predictions_file = 'textcnn/train_logs/predictions.csv'
labels_file = 'data/textcnn/labels.json'

def sparse_predictions_to_dense(sparse_predictions, num_classes):
    dense_predictions = np.zeros([num_classes])

    for i in range(int(len(sparse_predictions) / 2)):
        label = int(sparse_predictions[i * 2])
        score = float(sparse_predictions[i * 2 + 1])
        dense_predictions[label] = score

    return dense_predictions

def sparse_labels_to_dense(sparse_labels, num_classes):
    dense_labels = np.zeros([num_classes], dtype=int)

    for label in sparse_labels:
        dense_labels[label] = 1

    return dense_labels


with open(labels_file) as f:
    labels = json.load(f)

eval = eval_util.EvaluationMetrics(4716, 20)

count = 0
batch_num = 4096
batch_predictions = []
batch_labels = []
with open(predictions_file) as f:
    reader = csv.DictReader(f)
    for row in reader:
        sparse_predictions = row['LabelConfidencePairs'].split(' ')
        sparse_labels = labels[row['VideoId']]

        dense_predictions = sparse_predictions_to_dense(sparse_predictions,
                                                        4716)
        dense_labels = sparse_labels_to_dense(sparse_labels, 4716)

        batch_predictions.append(dense_predictions)
        batch_labels.append(dense_labels)
        count += 1

        batch_num -= 1
        if batch_num == 0:
            batch_predictions = np.asarray(batch_predictions)
            batch_labels = np.asarray(batch_labels)
            eval.accumulate(batch_predictions, batch_labels, 0)
            batch_num = 4096
            batch_predictions = []
            batch_labels = []
            print('Count: {}, GAP: {}'.format(count, eval.get()['gap']))

if len(batch_predictions) != 0:
    batch_predictions = np.asarray(batch_predictions)
    batch_labels = np.asarray(batch_labels)
    eval.accumulate(batch_predictions, batch_labels, 0)

print('GAP: {}'.format(eval.get()['gap']))
