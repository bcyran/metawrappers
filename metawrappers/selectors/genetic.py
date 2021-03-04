from operator import itemgetter

import numpy as np

from metawrappers.base import WrapperSelector
from metawrappers.common.mask import random_flip, random_mask
from metawrappers.common.run_time import RunTimeMixin
from metawrappers.common.utils import pairwise


class GASelector(WrapperSelector, RunTimeMixin):
    """Genetic Algorithm feature selector.

    Parameters
    ----------
    estimator : ``Estimator`` instance
        A supervised learning estimator with a ``fit`` method.
    population_size : int, default=10
        Size of the population.
    elite_size : int, default=2
        Number of best individuals from each generation carried over unaltered to the next one.
    mutation_rate : float, default=0.2
        Probability of mutations occuring.
    iterations : int, default=20
        The number of iterations to perform.
    run_time : int, default=None
        Maximum runtime of the selector in milliseconds. If set supersedes the ``iterations`` param.
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
        population_size=15,
        elite_size=2,
        mutation_rate=0.2,
        iterations=20,
        run_time=None,
        scoring="accuracy",
        cv=5,
        n_jobs=-1,
        random_state=None,
    ):
        super().__init__(estimator, scoring, cv, n_jobs, random_state)
        self.iterations = iterations
        self.run_time = run_time
        self.population_size = population_size
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate
        self._population = None
        self._population_with_fitnesses = None
        self._mating_pool = None

    def _select_features(self, X, y):
        self._start_timer()
        iterations = 0

        self._initialize(X, y)
        best_mask, best_fitness = self._population_with_fitnesses[0]

        while not self._should_end(iterations):
            self._selection()
            self._breeding()
            self._mutation()
            self._evaluate_population(X, y)

            if self._population_with_fitnesses[0][1] > best_fitness:
                best_mask, best_fitness = self._population_with_fitnesses[0]

            iterations += 1

        return best_mask

    def _initialize(self, X, y):
        self._population = list(
            random_mask(X.shape[1], random_state=self._rng) for _ in range(self.population_size)
        )
        self._evaluate_population(X, y)

    def _selection(self):
        masks, fitnesses = zip(*self._population_with_fitnesses)
        masks, fitnesses = np.array(masks), np.array(fitnesses)
        selection_probs = fitnesses / np.sum(fitnesses)
        n_select = self.population_size - self.elite_size + 1
        selected = self._rng.choice(self.population_size, size=n_select, p=selection_probs)
        self._mating_pool = masks[selected]

    def _breeding(self):
        elite = self._population_with_fitnesses[: self.elite_size]
        self._population = list(map(itemgetter(0), elite))
        for parent1, parent2 in pairwise(self._mating_pool):
            self._population.append(self._sp_crossover(parent1, parent2))

    def _sp_crossover(self, parent1, parent2):
        # We need to ensure that at lest one feature is selected so we always cut
        # after the first `True` in parent1. If it's on the last position then we switch parents.
        select_start = np.argmax(parent1) + 1
        if select_start == len(parent1):
            parent1, parent2 = parent2, parent1
            select_start = np.argmax(parent1) + 1
        crossover_point = self._rng.randint(select_start, len(parent1))
        return np.hstack((parent1[0:crossover_point], parent2[crossover_point:]))

    def _mutation(self):
        for i in range(len(self._population)):
            if self._rng.random() <= self.mutation_rate:
                self._population[i] = random_flip(self._population[i], random_state=self._rng)

    def _evaluate_population(self, X, y):
        self._population_with_fitnesses = [
            (individual, self._fitness(individual, X, y)) for individual in self._population
        ]
        self._population_with_fitnesses.sort(key=itemgetter(1), reverse=True)
