from metawrappers import HCSelector


# fmt: off
EXPECTED_SUPPORT = [False, False, False, False, False, False, False, False, True, False, True, True,
                    False, True, False, False, True, False, False, False, False, True, True, False,
                    False, True, False, True, False, False]
# fmt: on
EXPECTED_SCORE = 0.9402460456942003


def test_hc_selector(classifier, dataset, random_state):
    X, y = dataset
    selector = HCSelector(classifier, random_state=random_state)
    X_r = selector.fit_transform(X, y)
    classifier.fit(X_r, y)
    assert classifier.score(X_r, y) == EXPECTED_SCORE
    assert list(selector.support_) == EXPECTED_SUPPORT
