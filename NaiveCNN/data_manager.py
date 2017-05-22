import os
import numpy as np
import _pickle as cPickle

from utils import cifar_rgb_to_grayscale

def load_data(path):
    """
        Description: Load the CIFAR dataset for a repository
    """

    # Our two lists, containing the training and testing data
    train_data = []
    train_labels = []
    test_data = []
    test_labels = []
    
    # Since not everything is in one file...
    for f in os.listdir(path):
        if 'test' in f:
            test_data, test_labels = _load_file(os.path.join(path, f))
        elif 'data' in f:
            print(f)
            sub_train_data, sub_train_labels = _load_file(os.path.join(path, f))
            train_data.extend(sub_train_data)
            train_labels.extend(sub_train_labels)
        else:
            continue

    return (
        [cifar_rgb_to_grayscale(row) for row in train_data],
        train_labels,
        [cifar_rgb_to_grayscale(row) for row in test_data],
        test_labels
    )

def _load_file(path):
    """
        Description: Simple function that loads the CIFAR data as described on the
                     official website.
    """

    with open(path, 'rb') as f:
        d = cPickle.load(f, encoding='bytes')
    return d.get(b'data'), d.get(b'labels')

def save_data(path, weights):
    """
        Description: Simple function to write the weights to a file.
    """

    with open(path, 'rb') as f:
        f.write(weights)
