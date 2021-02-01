from collections import defaultdict
from statistics import mean

import pandas as pd
from scipy.stats import wilcoxon
from sklearn.model_selection import StratifiedKFold


class Comparator:
    """Performs multiple cross-validations using multiple estimators in order to compare them.

    Parameters
    ----------
    estimators : list of ``Estimator`` instances
        Estimators to compare.
    n_tests : int, default=25
        Number of cross-validations to perform.
    n_splits : int, default=5
        Number of folds in cross-validation.
    random_state : int, ``RandomState`` instance or None, default=None
        Controls randomness of the selector. Pass an int for reproducible output across multiple
        function calls.
    verbose : bool default=True
        Whether to print progress log and report.

    Attributes
    ----------
    results : ``pandas.DataFrame``
        DataFrame containing results for each test for each estimator.
    """

    def __init__(self, estimators, n_tests=25, n_splits=5, random_state=None, verbose=True):
        self.estimators = estimators
        self.n_tests = n_tests
        self.n_splits = n_splits
        self.random_state = random_state
        self.verbose = verbose
        self.results = pd.DataFrame(columns=self.estimator_names)

    @property
    def estimator_names(self):
        return [e.__class__.__name__ for e in self.estimators]

    def run(self, X, y):
        """Run the tests on the given dataset.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples of dataset to test on.
        y : array-like of shape (n_samples,)
            The target values of dataset to test on.
        """
        for i in range(self.n_tests):
            if self.verbose:
                print(f"Running test {i + 1}/{self.n_tests}: ", end="")

            self._run_test(X, y)

        if self.verbose:
            print("Comparison summary:")
            print(self.summary())

    def _run_test(self, X, y):
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)

        scores = defaultdict(list)
        for train, test in skf.split(X, y):
            for estimator in self.estimators:
                estimator.fit(X[train], y[train])
                score = estimator.score(X[test], y[test])
                scores[estimator.__class__.__name__].append(score)

                if self.verbose:
                    print(".", end="", flush=True)

            if self.verbose:
                print("/", end="", flush=True)

        if self.verbose:
            print("")

        self._append_to_results(scores)

    def _append_to_results(self, scores):
        averages = {}
        for estimator_name, estimator_scores in scores.items():
            averages[estimator_name] = mean(estimator_scores)

        self.results = self.results.append(averages, ignore_index=True)

    def summary(self, baseline=0, significance_level=0.05):
        """Return summary of comparison results.

        Attributes
        ----------
        baseline : int or str, default=0
            Index or name of baseline estimator. The first given is the default.
        significance_level : float, default=0.05
            Significance level for decision whether H0 can be rejected.

        Returns
        -------
        summary : ``pandas.DataFrame``
            DataFrame containing average score and Wilcoxon signed-rank test results
            with respect to baseline for each estimator.
        """
        baseline_name = self.estimator_names[baseline] if isinstance(baseline, int) else baseline
        averages = self.results.mean()

        summary = pd.DataFrame(columns=["avg", "W", "p", "h0 rejected"])
        for estimator_name in self.results.columns:
            if estimator_name == baseline_name:
                w, p = None, None
            else:
                w, p = wilcoxon(self.results[baseline_name], self.results[estimator_name])
            summary.loc[estimator_name] = [averages[estimator_name], w, p, None]

        summary["h0 rejected"] = summary["p"] < significance_level
        return summary
