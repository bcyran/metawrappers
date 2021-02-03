import numpy as np
from sklearn.utils import check_random_state


def random_mask(n_features, min_select, max_select, random_state=None):
    rng = check_random_state(random_state)
    n_select = rng.randint(min_select, max_select)
    mask = np.array([False] * n_features)
    mask[rng.choice(n_features, n_select, replace=False)] = True
    return mask


def random_neighbor(mask, min_select, max_select, random_state=None):
    rng = check_random_state(random_state)
    if mask.sum() == min_select:
        index_map = np.where(np.logical_not(mask))[0]
    elif mask.sum() == max_select:
        index_map = np.where(mask)[0]
    else:
        index_map = np.arange(mask.shape[0])
    flip_index = index_map[rng.randint(0, index_map.size)]
    mask[flip_index] = not mask[flip_index]
    return mask
