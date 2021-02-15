import time
from abc import ABCMeta, abstractmethod

import numpy as np
from sklearn.base import BaseEstimator, MetaEstimatorMixin, clone
from sklearn.feature_selection import SelectorMixin
from sklearn.metrics import get_scorer
from sklearn.model_selection import cross_val_score
from sklearn.utils import check_random_state
from sklearn.utils.metaestimators import if_delegate_has_method
from sklearn.utils.validation import check_is_fitted

from metawrappers.utils import adjust_feature_bounds


class WrapperSelector(SelectorMixin, MetaEstimatorMixin, BaseEstimator, metaclass=ABCMeta):
    """Base class for all feature selectors.

    Parameters
    ----------
    estimator : ``Estimator`` instance
        A supervised learning estimator with a ``fit`` method.
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
        min_features=1,
        max_features=-1,
        scoring="accuracy",
        cv=5,
        n_jobs=-1,
        random_state=None,
    ):
        self.estimator = estimator
        self.min_features = min_features
        self.max_features = max_features
        self.scoring = scoring
        self.scorer = get_scorer(scoring)
        self.cv = cv
        self.n_jobs = n_jobs
        self.random_state = random_state
        self._rng = check_random_state(random_state)
        # Real values after adjusting
        self._min_features = None
        self._max_features = None
        self._start_time = 0

    @abstractmethod
    def _select_features(self, X, y):
        """Perform the feature selection.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples.
        y : array-like of shape (n_samples,)
            The target values.

        Returns
        -------
        mask : ndarray of shape (n_features,)
            The mask of selected features.
        """

    def _score_mask(self, mask, X, y):
        """Evaluate score of the given feature mask.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples.
        y : array-like of shape (n_samples,)
            The target values.

        Returns
        -------
        score : float
            The score of the given feature mask.
        """
        X = X[:, mask]
        if self.cv:
            return np.average(cross_val_score(self.estimator, X, y, n_jobs=self.n_jobs))
        else:
            self.estimator.fit(X, y)
            return self.scorer(self.estimator, X, y)

    @property
    def classes_(self):
        return self.estimator_.classes_

    def fit(self, X, y):
        """Fit the model and then the underlying estimator on the selected features.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples.
        y : array-like of shape (n_samples,)
            The target values.
        """
        self._min_features, self._max_features = adjust_feature_bounds(
            self.min_features, self.max_features, X
        )
        support_ = self._select_features(X, y)
        features = np.arange(X.shape[1])[support_]
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X[:, features], y)
        self.n_features_ = support_.sum()
        self.support_ = support_
        return self

    def _get_support_mask(self):
        check_is_fitted(self)
        return self.support_

    def _start_timer(self):
        self._start_time = time.perf_counter_ns()

    def _time(self):
        return time.perf_counter_ns() - self._start_time

    def _time_ms(self):
        return self._time() // (10 ** 6)

    def _should_end(self, iterations=None):
        if self.run_time:
            if self._time_ms() >= self.run_time:
                return True
        elif iterations:
            if iterations >= self.iterations:
                return True

    @if_delegate_has_method(delegate="estimator")
    def predict(self, X):
        """Predict using the underlying estimator and selected set of features.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : array of shape (n_samples,)
            The predicted target values.
        """
        check_is_fitted(self)
        return self.estimator_.predict(self.transform(X))

    @if_delegate_has_method(delegate="estimator")
    def score(self, X, y):
        """Predict using the underlying estimator and selected set of features, return the
        score of the underlying estimator.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples.
        y : array-like of shape (n_samples,)
            The target values.

        Returns
        -------
        score : float
            Mean accuracy of prediction on given data.
        """
        check_is_fitted(self)
        return self.estimator_.score(self.transform(X), y)

    @if_delegate_has_method(delegate="estimator")
    def decision_function(self, X):
        """Compute the decision function of X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        dec : array, shape = (n_samples, n_classes) or (n_samples)
            The decision function of the input samples.
        """
        check_is_fitted(self)
        return self.estimator_.decision_function(self.transform(X))

    @if_delegate_has_method(delegate="estimator")
    def predict_proba(self, X):
        """Predict class probabilities for X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        p : array of shape (n_samples, n_classes)
            The class probabilities of the input samples.
        """
        check_is_fitted(self)
        return self.estimator_.predict_proba(self.transform(X))

    @if_delegate_has_method(delegate="estimator")
    def predict_log_proba(self, X):
        """Predict class log-probabilities for X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        p : array of shape (n_samples, n_classes)
            The class log-probabilities of the input samples.
        """
        check_is_fitted(self)
        return self.estimator_.predict_log_proba(self.transform(X))
