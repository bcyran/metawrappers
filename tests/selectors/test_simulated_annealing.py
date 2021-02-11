from metawrappers import SASelector


# fmt: off
EXPECTED_SUPPORT = [True, True, True, False, False, True, False, True, True, False, True, True,
                    False, True, True, True, False, True, False, True, False, True, True, False,
                    False, True, False, True, False, False]
# fmt: on
EXPECTED_SCORE = 0.9367311072056239


def test_sa_selector(classifier, dataset, random_state):
    X, y = dataset
    selector = SASelector(classifier, random_state=random_state)
    X_r = selector.fit_transform(X, y)
    classifier.fit(X_r, y)
    assert classifier.score(X_r, y) == EXPECTED_SCORE
    assert list(selector.support_) == EXPECTED_SUPPORT
