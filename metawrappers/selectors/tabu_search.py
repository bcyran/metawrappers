from collections import deque
from operator import itemgetter

import numpy as np

from metawrappers.base import WrapperSelector
from metawrappers.common.local_search import LSMixin
from metawrappers.common.mask import flip_neighborhood
from metawrappers.common.run_time import RunTimeMixin


class TSSelector(WrapperSelector, LSMixin, RunTimeMixin):
    """Tabu Search with limited first-improvement strategy.

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
    score_neighbors : int, default=10
        Maximum number of neighbors to score in each iteration.
    reset_threshold : int or None, default=None
        Number of non-improving iterations after which search is reinitialized.
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
        score_neighbors=10,
        reset_threshold=None,
        scoring="accuracy",
        cv=5,
        n_jobs=-1,
        random_state=None,
    ):
        super().__init__(estimator, scoring, cv, n_jobs, random_state)
        self.iterations = iterations
        self.run_time = run_time
        self.tabu_length = tabu_length
        self.score_neighbors = score_neighbors
        self.reset_threshold = reset_threshold
        self._tabu_list = None

    def _select_features(self, X, y):
        self._start_timer()
        self._tabu_list = deque(maxlen=self.tabu_length)
        iterations = non_improving_iterations = 0

        cur_mask, cur_score = self._random_mask_with_score(X, y)
        best_mask, best_score = cur_mask, cur_score

        while not self._should_end(iterations):
            cur_mask, cur_score = self._best_neighbor_with_score(cur_mask, X, y, best_score)

            if cur_score > best_score:
                best_mask, best_score = cur_mask, cur_score
                non_improving_iterations = 0
            else:
                non_improving_iterations += 1

            self._tabu_list.append(cur_mask)

            if self.reset_threshold and non_improving_iterations >= self.reset_threshold:
                cur_mask, cur_score = self._random_mask_with_score(X, y)
                non_improving_iterations = 0

            iterations += 1

        return best_mask

    def _best_neighbor_with_score(self, mask, X, y, best_score):
        neighbors = flip_neighborhood(mask)
        self._rng.shuffle(neighbors)

        scored_neighbors = []
        for mask in neighbors:
            if self._is_tabu(mask):
                continue

            score = self._fitness(mask, X, y)
            if score > best_score:
                return mask, score

            scored_neighbors.append((mask, score))

            if len(scored_neighbors) >= self.score_neighbors:
                break

        return max(scored_neighbors, key=itemgetter(1))

    def _is_tabu(self, mask):
        return any(np.array_equal(mask, tabu) for tabu in self._tabu_list)
