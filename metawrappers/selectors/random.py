from metawrappers.base import WrapperSelector
from metawrappers.utils import random_mask


class RandomSelector(WrapperSelector):
    """Random feature selector. Used as a baseline metric in comparisons."""

    def _select_features(self, X, y):
        return random_mask(X.shape[1], self.n_features_to_select)
