import numpy as np


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


def tanh(x):
    return np.tanh(x)


def relu(x):
    return np.maximum(0., x)


def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1).reshape(x.shape[0], 1))
    return e_x / np.sum(e_x, axis=1).reshape(e_x.shape[0], 1)


def linear(x):
    return x
