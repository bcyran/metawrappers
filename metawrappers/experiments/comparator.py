from statistics import mean

import pandas as pd
from scipy.stats import wilcoxon
from sklearn.model_selection import StratifiedKFold


def _check_estimators(estimators):
    if isinstance(estimators, dict):
        return estimators
    elif isinstance(estimators, (list, tuple)):
        labels = [e.__class__.__name__ for e in estimators]
        if len(set(labels)) != len(labels):
            raise RuntimeError("non-unique estimators given, use dict to label them")
        return dict(zip(labels, estimators))
    else:
        raise RuntimeError("'estimators' argument must be one of: dict, list, tuple")


class Comparator:
    """Performs multiple cross-validations using multiple estimators in order to compare them.

    Parameters
    ----------
    estimators : {dict, list, tuple}
        Either dict mapping labels to estimators or list of estimators.
        If list is given, class names will be used as labels, thus estimator classes must be unique.
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
        self._estimators = _check_estimators(estimators)
        self.n_tests = n_tests
        self.n_splits = n_splits
        self.random_state = random_state
        self.verbose = verbose
        columns_index = pd.MultiIndex.from_product(
            [self._estimators.keys(), ["#feat", "score"]], names=["Estimator", "Value"]
        )
        self.results = pd.DataFrame(columns=columns_index)

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

        run_results = {name: {"features": [], "scores": []} for name in self._estimators.keys()}
        for train, test in skf.split(X, y):
            for label, estimator in self._estimators.items():
                estimator.fit(X[train], y[train])
                score = estimator.score(X[test], y[test])
                run_results[label]["scores"].append(score)
                run_results[label]["features"].append(getattr(estimator, "n_features_", X.shape[1]))

                if self.verbose:
                    print(".", end="", flush=True)

            if self.verbose:
                print("/", end="", flush=True)

        if self.verbose:
            print("")

        self._append_to_results(run_results)

    def _append_to_results(self, run_results):
        row = []
        for name, results_dict in run_results.items():
            row.append(mean(results_dict["features"]))
            row.append(mean(results_dict["scores"]))

        self.results = self.results.append(
            pd.Series(row, index=self.results.columns), ignore_index=True
        )

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
        summary = pd.DataFrame(columns=["#feat", "avg score", "max score", "W", "p", "h0 rejected"])

        baseline_name = (
            list(self._estimators.keys())[baseline] if isinstance(baseline, int) else baseline
        )
        averages = self.results.mean()

        for estimator in self._estimators.keys():
            if estimator == baseline_name:
                w, p = None, None
            else:
                w, p = wilcoxon(
                    self.results[baseline_name]["score"], self.results[estimator]["score"]
                )

            max_score = self.results[estimator]["score"].max()
            summary.loc[estimator] = [*averages[estimator], max_score, w, p, None]

        summary["h0 rejected"] = summary["p"] < significance_level
        return summary
