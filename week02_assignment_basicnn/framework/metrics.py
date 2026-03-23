import numpy as np


def accuracy(y_pred, y_true):

    preds = np.argmax(y_pred, axis=1)
    labels = np.argmax(y_true, axis=1)

    return np.mean(preds == labels)


def binary_accuracy(y_pred, y_true):

    preds = (y_pred > 0.5).astype(int)

    return np.mean(preds == y_true)