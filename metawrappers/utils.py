import numpy as np


def random_mask(n_features, n_select):
    mask = np.array([False] * n_features)
    mask[np.random.choice(n_features, n_select, replace=False)] = True
    return mask


def random_neighbor(mask):
    true_map = np.where(mask)[0]
    false_map = np.where(np.logical_not(mask))[0]
    mask[true_map[np.random.randint(0, true_map.size)]] = False
    mask[false_map[np.random.randint(0, false_map.size)]] = True
    return mask
