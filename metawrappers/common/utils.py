import numpy as np


def sigmoid(val):
    return 1 / (1 + np.exp(-val))
