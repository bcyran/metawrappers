from metawrappers import HCSelector


# fmt: off
EXPECTED_SUPPORT = [False, True, False, False, False, False, True, False, True, False, True, False,
                    False, False, False, False, True, True, False, True, False, False, True, True,
                    False, False, False, False, True, True]
# fmt: on
EXPECTED_SCORE = 0.9209138840070299


def test_hc_selector(classifier, dataset, random_state):
    X, y = dataset
    selector = HCSelector(classifier, random_state=random_state)
    X_r = selector.fit_transform(X, y)
    classifier.fit(X_r, y)
    assert classifier.score(X_r, y) == EXPECTED_SCORE
    assert list(selector.support_) == EXPECTED_SUPPORT