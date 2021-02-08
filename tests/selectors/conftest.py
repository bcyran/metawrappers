import pytest
from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC


@pytest.fixture(scope="function")
def classifier():
    return SVC()


@pytest.fixture(scope="session")
def dataset():
    return load_breast_cancer(return_X_y=True)
