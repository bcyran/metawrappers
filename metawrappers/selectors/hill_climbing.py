from metawrappers.base import WrapperSelector
from metawrappers.common.local_search import LSMixin
from metawrappers.common.run_time import RunTimeMixin


class HCSelector(WrapperSelector, LSMixin, RunTimeMixin):
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
    reset_threshold : int or None, default=None
        Number of non-improving iterations after which search is reinitialized.
    feature_num_penalty : float, default=0
        Controls how much number of selected features affects the fitness measure.
        Increasing this number will push the selector to minimize feature number.
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
        reset_threshold=None,
        feature_num_penalty=0,
        scoring="accuracy",
        cv=5,
        n_jobs=-1,
        random_state=None,
    ):
        super().__init__(estimator, feature_num_penalty, scoring, cv, n_jobs, random_state)
        self.iterations = iterations
        self.run_time = run_time
        self.neighborhood = neighborhood
        self.reset_threshold = reset_threshold

    def _select_features(self, X, y):
        self._start_timer()
        iterations = non_improving_iterations = 0

        cur_mask, cur_fitness = self._random_mask_with_fitness(X, y)
        best_mask, best_fitness = cur_mask, cur_fitness

        while not self._should_end(iterations):
            next_mask, next_fitness = self._random_neighbor_with_fitness(cur_mask, X, y)

            if next_fitness > cur_fitness:
                cur_mask, cur_fitness = next_mask, next_fitness
                if cur_fitness > best_fitness:
                    best_mask, best_fitness = cur_mask, cur_fitness
                non_improving_iterations = 0
            else:
                non_improving_iterations += 1

            if self.reset_threshold and non_improving_iterations >= self.reset_threshold:
                cur_mask, cur_fitness = self._random_mask_with_fitness(X, y)
                non_improving_iterations = 0

            iterations += 1

        return best_mask
