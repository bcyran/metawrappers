from metawrappers.base import WrapperSelector
from metawrappers.common.mask import random_mask, random_neighbor


class HCSelector(WrapperSelector):
    """Hill Climbing feature selector.

    Parameters
    ----------
    estimator : ``Estimator`` instance
        A supervised learning estimator with a ``fit`` method.
    iterations : int, default=50
        The number of iterations to perform.
    run_time : int, default=None
        Maximum runtime of the selector in milliseconds. If set supersedes the ``iterations`` param.
    neighborhood: {"1-flip", "2-flip"}, default=None
        Type of the neighborhood. `None` will choose randomly every time neighbor is requested.
    min_features : int, default=1
        The minimal number of features to select.
    max_features : int, default=-1
        The maxmimal number of features to select. -1 means all features.
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
        *,
        iterations=50,
        run_time=None,
        neighborhood="2-flip",
        min_features=1,
        max_features=-1,
        scoring="accuracy",
        cv=5,
        n_jobs=-1,
        random_state=None,
    ):
        super().__init__(estimator, min_features, max_features, scoring, cv, n_jobs, random_state)
        self.iterations = iterations
        self.run_time = run_time
        self.neighborhood = neighborhood

    def _select_features(self, X, y):
        self._start_timer()
        iteration = 1

        cur_mask = random_mask(X.shape[1], self._min_features, self._max_features, self._rng)
        cur_score = self._score_mask(cur_mask, X, y)

        while True:
            next_mask = random_neighbor(
                self.neighborhood, cur_mask, self._min_features, self._max_features, self._rng
            )
            next_score = self._score_mask(next_mask, X, y)

            if next_score > cur_score:
                cur_mask, cur_score = next_mask, next_score

            if self._end_condition(iteration):
                break
            iteration += 1

        return cur_mask
