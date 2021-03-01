from metawrappers import SASelector
from tests.selectors.conftest import assert_selector_result


# fmt: off
EXPECTED_SUPPORT = [True, True, True, False, False, True, False, True, True, False, True, True,
                    False, True, True, True, False, True, False, True, False, True, True, False,
                    False, True, False, True, False, False]
# fmt: on
EXPECTED_SCORE = 0.9367311072056239


def test_sa_selector(classifier, dataset, random_state):
    assert_selector_result(
        SASelector(classifier, random_state=random_state),
        classifier,
        dataset,
        EXPECTED_SUPPORT,
        EXPECTED_SCORE,
    )
