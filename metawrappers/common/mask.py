import numpy as np
from sklearn.utils import check_random_state


def random_mask(n_features, min_select, max_select, random_state=None):
    rng = check_random_state(random_state)
    n_select = rng.randint(min_select, max_select)
    mask = np.array([False] * n_features)
    mask[rng.choice(n_features, n_select, replace=False)] = True
    return mask


def flip_neighbor(mask, min_select, max_select, random_state=None):
    rng = check_random_state(random_state)
    allowed_indices = allowed_flips(mask, min_select, max_select)
    flip_index = rng.choice(allowed_indices, 1)[0]
    return flip(mask, flip_index)


def two_flip_neigbor(mask, min_select, max_select, random_state=None):
    rng = check_random_state(random_state)
    first_flip_allowed = allowed_flips(mask, min_select, max_select)
    first_flip_index = rng.choice(first_flip_allowed, 1)[0]
    mask = flip(mask, first_flip_index)
    second_flip_allowed = allowed_flips(mask, min_select, max_select)
    second_flip_allowed = second_flip_allowed[second_flip_allowed != first_flip_index]
    second_flip_index = rng.choice(second_flip_allowed, 1)[0]
    return flip(mask, second_flip_index)


def flip_neighborhood(mask, min_select, max_select):
    return [flip(mask, index) for index in allowed_flips(mask, min_select, max_select)]


def allowed_flips(mask, min_select, max_select):
    if mask.sum() == min_select:
        return np.where(np.logical_not(mask))[0]
    elif mask.sum() == max_select:
        return np.where(mask)[0]
    return np.arange(mask.shape[0])


def flip(mask, index):
    new_mask = mask.copy()
    new_mask[index] = not new_mask[index]
    return new_mask


NEIGHBORHOOD_DICT = {
    "1-flip": flip_neighbor,
    "2-flip": two_flip_neigbor,
}


def get_neighbor(neighborhood, mask, min_select, max_select, random_state=None):
    if not neighborhood:
        rng = check_random_state(random_state)
        neighborhood = rng.choice(list(NEIGHBORHOOD_DICT.keys()), 1)[0]
    return NEIGHBORHOOD_DICT[neighborhood](mask, min_select, max_select, random_state)
