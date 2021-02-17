import numpy as np

from metawrappers.base import WrapperSelector
from metawrappers.common.mask import random_mask
from metawrappers.common.run_time import RunTimeMixin
from metawrappers.common.utils import sigmoid


V_CLAMP = (-5, 5)


class Particle:
    def __init__(self, position, velocity):
        self.position = position
        self._score = 0
        self.best_position = position
        self.best_score = 0
        self.velocity = velocity

    @property
    def score(self):
        return self._score

    @score.setter
    def score(self, score):
        self._score = score
        if score > self.best_score:
            self.best_score = score


class PSOSelector(WrapperSelector, RunTimeMixin):
    """Particle Swarm Optimization selector.

    Parameters
    ----------
    estimator : ``Estimator`` instance
        A supervised learning estimator with a ``fit`` method.
    n_particles : int, default=5
        Number of particles in the swarm.
    inertia : float, default=0.5
        Particles inertia.
    cognitive : float, default=1
        Particles cognitive velocity coefficient.
    social : float, default=2
        Particles social velocity coefficient.
    iterations : int, default=20
        The number of iterations to perform.
    run_time : int, default=None
        Maximum runtime of the selector in milliseconds. If set supersedes the ``iterations`` param.
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
        n_particles=5,
        inertia=0.5,
        cognitive=1,
        social=2,
        iterations=20,
        run_time=None,
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
        self.n_particles = n_particles
        self.inertia = inertia
        self.cognitive = cognitive
        self.social = social
        self._swarm = []

    def _select_features(self, X, y):
        self._start_timer()
        self._init_swarm(X.shape[1])
        iterations = 0

        self._update_scores(X, y)
        best_mask, best_score = self._get_best()

        while not self._should_end(iterations):
            self._update_velocities(best_mask)
            self._update_positions()
            self._update_scores(X, y)
            best_mask, best_score = self._get_best()
            iterations += 1

        return best_mask

    def _init_swarm(self, n_features):
        self._swarm = []
        for _ in range(self.n_particles):
            mask = random_mask(n_features, self._min_features, self._max_features, self._rng)
            velocity = self._rng.uniform(-1, 1, n_features)
            self._swarm.append(Particle(mask, velocity))

    def _update_scores(self, X, y):
        for particle in self._swarm:
            particle.score = self._score_mask(particle.position, X, y)

    def _update_velocities(self, global_best_position):
        for particle in self._swarm:
            r1 = self._rng.uniform(0, 1)
            r2 = self._rng.uniform(0, 1)
            p_best_distance = particle.best_position.astype(int) - particle.position.astype(int)
            g_best_distance = global_best_position.astype(int) - particle.position.astype(int)
            v_cognitive = self.cognitive * r1 * p_best_distance
            v_social = self.social * r2 * g_best_distance
            velocity = self.inertia * particle.velocity + v_cognitive + v_social
            particle.velocity = np.clip(velocity, *V_CLAMP)

    def _update_positions(self):
        for particle in self._swarm:
            rand = self._rng.uniform(0, 1, particle.position.shape[0])
            particle.position = rand < sigmoid(-particle.velocity)
            # TODO: Respect min_features and max_features?

    def _get_best(self):
        best_mask, best_score = None, 0
        for particle in self._swarm:
            if particle.best_score > best_score:
                best_mask, best_score = particle.best_position, particle.best_score
        return best_mask, best_score
