from collections import deque
from functools import partial
from itertools import filterfalse

import numpy as np
from sklearn.utils.extmath import cartesian

from metawrappers.base import WrapperSelector
from metawrappers.common.local_search import LSMixin
from metawrappers.common.mask import flip_neighborhood
from metawrappers.common.run_time import RunTimeMixin


class LTSSelector(WrapperSelector, LSMixin, RunTimeMixin):
    """Learning Tabu Search feature selector.
    See: https://hal.archives-ouvertes.fr/hal-01370396/document.

    Parameters
    ----------
    estimator : ``Estimator`` instance
        A supervised learning estimator with a ``fit`` method.
    iterations : int, default=20
        The number of iterations to perform.
    run_time : int, default=None
        Maximum runtime of the selector in milliseconds. If set supersedes the ``iterations`` param.
    tabu_length : int, default=15
        Number of elements in the tabu list.
    evaporation_rate : int, default=0.9
        Rate at which trail values are decreasing with iterations.
    score_neighbors : int, default=15
        Number of neighbors to actually score in each iteration.
    reset_threshold : int or None, default=None
        Number of non-improving iterations after which search is reinitialized.
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
        iterations=20,
        run_time=None,
        tabu_length=15,
        evaporation_rate=0.9,
        score_neighbors=15,
        reset_threshold=None,
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
        self.tabu_length = tabu_length
        self.evaporation_rate = evaporation_rate
        self.score_neighbors = score_neighbors
        self.reset_threshold = reset_threshold
        self._tabu_list = deque(maxlen=self.tabu_length)
        self._trails = None

    def _select_features(self, X, y):
        self._start_timer()
        iterations = non_improving_iterations = 0

        self._trails = np.zeros((X.shape[1], X.shape[1]))
        cur_mask, cur_score = self._random_mask_with_score(X, y)
        best_mask, best_score = cur_mask, cur_score

        while not self._should_end(iterations):
            cur_mask = self._best_neighbor(cur_mask, X, y)
            cur_score = self._score_mask(cur_mask, X, y)

            if cur_score > best_score:
                best_mask, best_score = cur_mask, cur_score
                non_improving_iterations = 0
            else:
                non_improving_iterations += 1

            self._tabu_list.append(cur_mask)

            if self.reset_threshold and non_improving_iterations >= self.reset_threshold:
                cur_mask, cur_score = self._random_mask_with_score(X, y)
                non_improving_iterations = 0

            self._update_trails(best_mask, best_score)

            iterations += 1

        return best_mask

    def _best_neighbor(self, mask, X, y):
        neighbors = flip_neighborhood(mask, self._min_features, self._max_features)
        non_tabu = filterfalse(self._is_tabu, neighbors)
        estimate_func = partial(self._estimate_neighbor, mask)
        best_estimated = sorted(non_tabu, key=estimate_func, reverse=True)[: self.score_neighbors]
        return sorted(best_estimated, key=lambda m: self._score_mask(m, X, y), reverse=True)[0]

    def _is_tabu(self, mask):
        return any(np.array_equal(mask, tabu) for tabu in self._tabu_list)

    def _estimate_neighbor(self, mask, neighbor):
        diff_index = np.where(mask != neighbor)[0][0]
        return self._trails.sum(axis=0)[diff_index]

    def _update_trails(self, mask, score):
        self._trails *= self.evaporation_rate
        selected_idx = np.where(mask)[0]
        intersections_idx = cartesian((selected_idx, selected_idx))
        self._trails[tuple(np.array(intersections_idx).T)] += score
