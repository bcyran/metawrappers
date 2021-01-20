from metawrappers.base import WrapperSelector
from metawrappers.utils import random_mask, random_neighbor


class HCSelector(WrapperSelector):
    """Hill Climbing feature selector.

    Parameters
    ----------
    estimator : ``Estimator`` instance
        A supervised learning estimator with a ``fit`` method.
    n_features_to_select : int, default=1
        The number of features to select.
    iterations : int, default=50
        The number of iterations to perform.
    scoring : str or callable, default='accuracy'
        Scoring metric to use for internal feature set evaluation. This and the following
        scoring-related attributes do not affect the `score` method.
        See `sklearn.metrics.get_scorer` documentation for more info.
    cv : int or callable, default=5
        Cross-validation to use for internal feature set evaluation.
        See `sklearn.model_selection.cross_val_score` documentation for more info.
    n_jobs : int, default=-1
        Number of CPU-s to use for internal feature set evaluation.
        See `sklearn.model_selection.cross_val_score` documentation for more info.
    random_state : int, ``RandomState`` instance or None, default=None
        Controls randomness of the selector. Pass an int for reproducible output across multiple
        function calls.

    Attributes
    ----------
    estimator_ : ``Estimator`` instance
        The fitted estimator used to select features.
    n_features_ : int
        The number of selected features.
    support_ : ndarray of shape (n_features,)
        The mask of selected features.
    """

    def __init__(
        self,
        estimator,
        n_features_to_select=1,
        iterations=50,
        scoring="accuracy",
        cv=5,
        n_jobs=-1,
        random_state=None,
    ):
        super().__init__(estimator, n_features_to_select, scoring, cv, n_jobs, random_state)
        self.iterations = iterations

    def _select_features(self, X, y):
        cur_mask = random_mask(X.shape[1], self.n_features_to_select, self._rng)
        cur_score = self._score_mask(cur_mask, X, y)
        best_mask, best_score = cur_mask, cur_score

        for i in range(self.iterations):
            cur_mask = random_neighbor(cur_mask, self._rng)
            cur_score = self._score_mask(cur_mask, X, y)

            if cur_score > best_score:
                best_mask, best_score = cur_mask, cur_score

        return best_mask
