from metawrappers import TSSelector


# fmt: off
EXPECTED_SUPPORT = [True, True, False, False, False, True, False, False, False, True, True, False,
                    False, False, False, False, True, False, False, False, False, True, True, False,
                    False, True, True, True, True, False]
# fmt: on
EXPECTED_SCORE = 0.9384885764499121


def test_ts_selector(classifier, dataset, random_state):
    X, y = dataset
    selector = TSSelector(classifier, random_state=random_state)
    X_r = selector.fit_transform(X, y)
    classifier.fit(X_r, y)
    assert classifier.score(X_r, y) == EXPECTED_SCORE
    assert list(selector.support_) == EXPECTED_SUPPORT
