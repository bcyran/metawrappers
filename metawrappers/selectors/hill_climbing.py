from metawrappers.base import WrapperSelector
from metawrappers.utils import random_mask, random_neighbor


class HCSelector(WrapperSelector):
    """Hill Climbing selector."""

    def __init__(self, *args, **kwargs):
        self.iterations = kwargs.pop("iterations", 50)
        super().__init__(*args, **kwargs)

    def _select_features(self, X, y):
        cur_mask = random_mask(X.shape[1], self.n_features_to_select)
        cur_score = self._score_mask(cur_mask, X, y)
        best_mask, best_score = cur_mask, cur_score

        for i in range(self.iterations):
            cur_mask = random_neighbor(cur_mask)
            cur_score = self._score_mask(cur_mask, X, y)

            if cur_score > best_score:
                best_mask, best_score = cur_mask, cur_score

        return best_mask
