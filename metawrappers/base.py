from abc import ABCMeta, abstractmethod

import numpy as np
from sklearn.base import BaseEstimator, MetaEstimatorMixin, clone
from sklearn.feature_selection import SelectorMixin
from sklearn.utils.metaestimators import if_delegate_has_method
from sklearn.utils.validation import check_is_fitted


class WrapperSelector(SelectorMixin, MetaEstimatorMixin, BaseEstimator, metaclass=ABCMeta):

    def __init__(self, estimator, *, n_features_to_select=None):
        self.estimator = estimator
        self.n_features_to_select = n_features_to_select

    @abstractmethod
    def _select_features(self, X, y):
        """Perform the feature selection.

        :param X: Training vectors.
        :param y: Target values.
        :return: Boolean mask of selected features.
        """

    @property
    def classes_(self):
        return self.estimator_.classes_

    def fit(self, X, y):
        """Perform feature selection and learn the estimator on the training data.

        :param X: Training vectors.
        :param y: Target values.
        """
        support_ = self._select_features(X, y)
        features = np.arange(X.shape[1])[support_]
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X[:, features], y)
        self.n_features_ = self.support_.sum()
        self.support_ = support_
        return self

    def _get_support_mask(self):
        check_is_fitted(self)
        return self.support_

    @if_delegate_has_method(delegate='estimator')
    def predict(self, X):
        """Predict using the underlying estimator and selected set of features.

        :param X: Training vectors.
        """
        check_is_fitted(self)
        return self.estimator_.predict(self.transform(X))

    @if_delegate_has_method(delegate='estimator')
    def score(self, X, y):
        """Predict using the underlying estimator and selected set of features, then return the
        score of the underlying estimator.

        :param X: Training vectors.
        """
        check_is_fitted(self)
        return self.estimator_.score(self.transform(X), y)

    @if_delegate_has_method(delegate='estimator')
    def decision_function(self, X):
        """Compute the decision function of X.

        :param X: Training vectors.
        """
        check_is_fitted(self)
        return self.estimator_.decision_function(self.transform(X))

    @if_delegate_has_method(delegate='estimator')
    def predict_proba(self, X):
        """Predict class probabilities for X.

        :param X: Training vectors.
        """
        check_is_fitted(self)
        return self.estimator_.predict_proba(self.transform(X))

    @if_delegate_has_method(delegate='estimator')
    def predict_log_proba(self, X):
        """Predict class log-probabilities for X.

        :param X: Training vectors.
        """
        check_is_fitted(self)
        return self.estimator_.predict_log_proba(self.transform(X))
