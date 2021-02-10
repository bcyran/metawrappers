import numpy as np
import pytest

from metawrappers import LTSSelector


EVAPORATION = 0.9


@pytest.fixture
def selector(classifier):
    selector = LTSSelector(classifier)
    selector.evaporation = EVAPORATION
    return selector


@pytest.mark.parametrize(
    "trails,mask,neighbor,expected_estimate",
    [
        (np.arange(9).reshape((3, 3)), np.array([1, 1, 0]), np.array([1, 0, 0]), 12),
        (np.arange(9).reshape((3, 3)), np.array([1, 1, 0]), np.array([1, 1, 1]), 15),
        (np.arange(9).reshape((3, 3)), np.array([0, 0, 0]), np.array([1, 0, 0]), 9),
    ],
)
def test_estimate_neighbor(selector, trails, mask, neighbor, expected_estimate):
    selector._trails = trails
    assert selector._estimate_neighbor(mask, neighbor) == expected_estimate


@pytest.mark.parametrize(
    "current_trails,mask,score,expected_trails",
    [
        (np.zeros((3, 3)), np.array([0, 1, 1]), 7, np.array([[0, 0, 0], [0, 7, 7], [0, 7, 7]])),
        (np.zeros((3, 3)), np.array([1, 0, 1]), 7, np.array([[7, 0, 7], [0, 0, 0], [7, 0, 7]])),
        (
            np.ones((3, 3)),
            np.array([1, 1, 0]),
            5,
            np.array([[5.9, 5.9, 0.9], [5.9, 5.9, 0.9], [0.9, 0.9, 0.9]]),
        ),
    ],
)
def test_update_trail(selector, current_trails, mask, score, expected_trails):
    selector._trails = current_trails.copy()
    selector._update_trails(mask, score)
    assert np.array_equal(selector._trails, expected_trails)
