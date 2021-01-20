import numpy as np
from sklearn.utils import check_random_state


def random_mask(n_features, n_select, random_state=None):
    rng = check_random_state(random_state)
    mask = np.array([False] * n_features)
    mask[rng.choice(n_features, n_select, replace=False)] = True
    return mask


def random_neighbor(mask, random_state=None):
    rng = check_random_state(random_state)
    true_map = np.where(mask)[0]
    false_map = np.where(np.logical_not(mask))[0]
    mask[true_map[rng.randint(0, true_map.size)]] = False
    mask[false_map[rng.randint(0, false_map.size)]] = True
    return mask
