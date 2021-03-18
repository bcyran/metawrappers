from operator import itemgetter

import numpy as np
from sklearn.feature_selection import f_classif

from metawrappers.base import WrapperSelector
from metawrappers.common.run_time import RunTimeMixin


class ACOSelector(WrapperSelector, RunTimeMixin):
    """Ant Colony Optimization feature selector.

    Parameters
    ----------
    estimator : ``Estimator`` instance
        A supervised learning estimator with a ``fit`` method.
    n_ants : int, default=5
        Number of ants.
    pheromone_coef : int, default=1
        Controls the importance of the pheromone value (commonly denoted as alpha).
    heuristic_coef : int, default=0.1
        Controls the importance of the heuristic desirability value (commonly denoted as beta).
    evaporation_rate : int, default=0.2
        The rate at which the pheromone value decreases (commonly denoted as rho).
    heuristic_func : callable, default=f_classif
        Function providing heuristic desirability information about features.
        This function should take two arrays: X and y, and return array of scores or a tuple:
        (scores, p-values).
    iterations : int, default=20
        The number of iterations to perform.
    run_time : int, default=None
        Maximum runtime of the selector in milliseconds. If set supersedes the ``iterations`` param.
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
        n_ants=10,
        pheromone_coef=1,
        heuristic_coef=0.1,
        evaporation_rate=0.2,
        heuristic_func=f_classif,
        iterations=20,
        run_time=None,
        feature_num_penalty=0,
        scoring="accuracy",
        cv=5,
        n_jobs=-1,
        random_state=None,
    ):
        super().__init__(estimator, feature_num_penalty, scoring, cv, n_jobs, random_state)
        self.iterations = iterations
        self.run_time = run_time
        self.n_ants = n_ants
        self.pheromone_coef = pheromone_coef
        self.heuristic_coef = heuristic_coef
        self.evaporation_rate = evaporation_rate
        self.heuristic_func = heuristic_func
        self._pheromone = None
        self._heuristic = None

    def _select_features(self, X, y):
        self._initialize(X, y)
        self._start_timer()
        iterations = 0

        best_mask, best_fitness = None, 0

        while not self._should_end(iterations):
            ants_with_fitnesses = self._ants_with_fitnesses(X, y)
            best_ant, best_ant_fitness = max(ants_with_fitnesses, key=itemgetter(1))

            if best_ant_fitness > best_fitness:
                best_mask, best_fitness = best_ant, best_ant_fitness

            self._update_pheromone(ants_with_fitnesses, best_fitness)

            iterations += 1

        return best_mask

    def _initialize(self, X, y):
        heuristic_scores = self.heuristic_func(X, y)
        if isinstance(heuristic_scores, (tuple, list)):
            heuristic_scores = heuristic_scores[0]
        np.nan_to_num(heuristic_scores, copy=False, nan=0, posinf=0, neginf=0)
        self._pheromone = np.ones((X.shape[1],))
        self._heuristic = heuristic_scores

    def _ants_with_fitnesses(self, X, y):
        ants = (self._construct_ant() for _ in range(self.n_ants))
        return list((ant, self._fitness(ant, X, y)) for ant in ants)

    def _construct_ant(self):
        effective_pheromone = self._pheromone ** self.pheromone_coef
        effective_heuristic = self._heuristic ** self.heuristic_coef
        feature_values = effective_pheromone * effective_heuristic
        feature_probs = feature_values / np.sum(feature_values)
        feature_num = self._rng.randint(1, feature_probs.size)
        selected = self._rng.choice(feature_probs.size, size=feature_num, p=feature_probs)
        ant = np.zeros(feature_probs.size, dtype=bool)
        ant[selected] = True
        return ant

    def _update_pheromone(self, ants_with_fitnesses, best_fitness):
        self._pheromone *= 1 - self.evaporation_rate
        for mask, fitness in ants_with_fitnesses:
            self._pheromone[mask] += 1 / (1 + (best_fitness - fitness) / best_fitness)
