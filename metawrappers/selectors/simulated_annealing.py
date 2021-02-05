from math import exp

from metawrappers.base import WrapperSelector
from metawrappers.common.mask import get_neighbor, random_mask


class SASelector(WrapperSelector):
    """Simulated Annealing feature selector.

    Parameters
    ----------
    estimator : ``Estimator`` instance
        A supervised learning estimator with a ``fit`` method.
    iterations : int, default=50
        The number of iterations to perform.
    neighborhood: {"1-flip", "2-flip"}, default=None
        Type of the neighborhood. `None` will choose randomly every time neighbor is requested.
    initial_temperature : int or float, default=10
        Initial annealing temperature.
    cooling_rate : float, default=0.05
        The factor by which the temperature will be decreased with each iteration.
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
        neighborhood="2-flip",
        initial_temperature=10,
        cooling_rate=0.05,
        min_features=1,
        max_features=-1,
        scoring="accuracy",
        cv=5,
        n_jobs=-1,
        random_state=None,
    ):
        super().__init__(estimator, min_features, max_features, scoring, cv, n_jobs, random_state)
        self.iterations = iterations
        self.neighborhood = neighborhood
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate

    def _select_features(self, X, y):
        temperature = self.initial_temperature
        cur_mask = random_mask(X.shape[1], self._min_features, self._max_features, self._rng)
        cur_score = self._score_mask(cur_mask, X, y)
        best_mask, best_score = cur_mask, cur_score

        for i in range(self.iterations):
            new_mask = get_neighbor(
                self.neighborhood, cur_mask, self._min_features, self._max_features, self._rng
            )
            new_score = self._score_mask(cur_mask, X, y)

            delta_score = new_score - cur_score

            if delta_score >= 0:
                cur_mask, cur_score = new_mask, new_score
                if new_score > best_score:
                    best_mask, best_score = new_mask, new_score
            elif exp(delta_score / temperature) > self._rng.random():
                cur_mask, cur_score = new_mask, new_score

            temperature *= 1 - self.cooling_rate

        return best_mask
