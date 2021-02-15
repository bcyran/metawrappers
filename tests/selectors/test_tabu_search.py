import numpy as np
import pytest

from metawrappers import LTSSelector


# fmt: off
EXPECTED_SUPPORT = [True, True, False, False, False, True, False, False, True, False, True, True,
                    False, True, False, False, True, True, False, False, False, False, True, False,
                    False, True, False, True, False, True]
# fmt: on
EXPECTED_SCORE = 0.9349736379613357


def test_lts_selector(classifier, dataset, random_state):
    X, y = dataset
    selector = LTSSelector(classifier, random_state=random_state)
    X_r = selector.fit_transform(X, y)
    classifier.fit(X_r, y)
    assert classifier.score(X_r, y) == EXPECTED_SCORE
    assert list(selector.support_) == EXPECTED_SUPPORT


@pytest.fixture
def selector(classifier):
    selector = LTSSelector(classifier, evaporation_rate=0.9)
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
