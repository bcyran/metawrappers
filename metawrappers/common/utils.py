from itertools import tee

import numpy as np


def sigmoid(val):
    return 1 / (1 + np.exp(-val))


def pairwise(iterable):
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)
