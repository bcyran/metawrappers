from metawrappers import PSOSelector


# fmt: off
EXPECTED_SUPPORT = [True, True, False, False, True, True, False, False, False, True, False, False,
                    False, False, True, False, True, False, False, False, True, True, True, False,
                    True, True, False, False, True, True]
# fmt: on
EXPECTED_SCORE = 0.9384885764499121


def test_pso_selector(classifier, dataset, random_state):
    X, y = dataset
    selector = PSOSelector(classifier, random_state=random_state)
    X_r = selector.fit_transform(X, y)
    classifier.fit(X_r, y)
    assert classifier.score(X_r, y) == EXPECTED_SCORE
    assert list(selector.support_) == EXPECTED_SUPPORT
