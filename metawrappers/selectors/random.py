from metawrappers.base import WrapperSelector
from metawrappers.utils import random_mask


class RandomSelector(WrapperSelector):
    """Random feature set selector. Used as a baseline metric in comparisons.

    Parameters
    ----------
    estimator : ``Estimator`` instance
        A supervised learning estimator with a ``fit`` method.
    n_features_to_select : int, default=1
        The number of features to select.
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

    def _select_features(self, X, y):
        return random_mask(X.shape[1], self.n_features_to_select, self._rng)
