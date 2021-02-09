import numpy as np
import pytest

from metawrappers.common.mask import allowed_flips, flip, flip_neighborhood


@pytest.mark.parametrize(
    "mask,index,expected_mask",
    [
        ([1, 1, 1, 1, 1], 0, [0, 1, 1, 1, 1]),
        ([0, 0, 0, 0, 0], 0, [1, 0, 0, 0, 0]),
        ([0, 1, 1, 0, 0], 2, [0, 1, 0, 0, 0]),
        ([0, 1, 1, 0, 0], 3, [0, 1, 1, 1, 0]),
    ],
)
def test_flip(mask, index, expected_mask):
    assert list(flip(np.array(mask), index)) == expected_mask


@pytest.mark.parametrize(
    "mask,min_select,max_select,expected_indices",
    [
        ([0, 0, 0, 0, 0], 0, 5, [0, 1, 2, 3, 4]),
        ([1, 0, 0, 0, 0], 1, 5, [1, 2, 3, 4]),
        ([0, 1, 1, 1, 1], 0, 4, [1, 2, 3, 4]),
        ([1, 0, 1, 0, 1], 1, 4, [0, 1, 2, 3, 4]),
        ([1, 0, 1, 0, 1], 3, 4, [1, 3]),
        ([1, 0, 1, 1, 1], 3, 4, [0, 2, 3, 4]),
    ],
)
def test_allowed_flips(mask, min_select, max_select, expected_indices):
    assert list(allowed_flips(np.array(mask), min_select, max_select)) == expected_indices


@pytest.mark.parametrize(
    "mask,min_select,max_select,expected_neighborhood",
    [
        (
            [0, 0, 0],
            0,
            3,
            [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ],
        ),
        (
            [1, 0, 0],
            0,
            3,
            [
                [0, 0, 0],
                [1, 1, 0],
                [1, 0, 1],
            ],
        ),
        (
            [1, 0, 0],
            0,
            1,
            [
                [0, 0, 0],
            ],
        ),
        (
            [1, 1, 0],
            1,
            2,
            [
                [0, 1, 0],
                [1, 0, 0],
            ],
        ),
    ],
)
def test_flip_neighborhood(mask, min_select, max_select, expected_neighborhood):
    neighborhood = []
    for neighbor in flip_neighborhood(np.array(mask), min_select, max_select):
        neighborhood.append(list(neighbor))
    assert neighborhood == expected_neighborhood
