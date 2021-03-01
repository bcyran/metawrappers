from metawrappers import RandomSelector
from tests.selectors.conftest import assert_selector_result


# fmt: off
EXPECTED_SUPPORT = [False, False, True, False, False, True, False, False, True, False, True, True,
                    False, True, False, False, True, True, False, False, False, False, True, True,
                    False, True, False, True, False, True]
# fmt: on
EXPECTED_SCORE = 0.9209138840070299


def test_random_selector(classifier, dataset, random_state):
    assert_selector_result(
        RandomSelector(classifier, random_state=random_state),
        classifier,
        dataset,
        EXPECTED_SUPPORT,
        EXPECTED_SCORE,
    )
