import pytest
from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC


@pytest.fixture(scope="function")
def classifier():
    return SVC()


@pytest.fixture(scope="session")
def dataset():
    return load_breast_cancer(return_X_y=True)


def assert_selector_result(selector, classifier, dataset, expected_support, expected_score):
    X, y = dataset
    X_r = selector.fit_transform(X, y)
    classifier.fit(X_r, y)
    assert classifier.score(X_r, y) == expected_score
    assert list(selector.support_) == expected_support
