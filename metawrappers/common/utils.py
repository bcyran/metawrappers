import numpy as np
from sklearn.utils.validation import check_random_state


def sigmoid(val):
    return 1 / (1 + np.exp(-val))


def roulette(probs, n_select, random_state):
    rng = check_random_state(random_state)
    probs_cumsum = np.cumsum(probs)
    return [np.searchsorted(probs_cumsum, rng.random()) for _ in range(n_select)]
