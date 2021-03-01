from metawrappers import TSSelector
from tests.selectors.conftest import assert_selector_result


# fmt: off
EXPECTED_SUPPORT = [True, True, False, False, False, True, False, False, False, True, True, False,
                    False, False, False, False, True, False, False, False, False, True, True, False,
                    False, True, True, True, True, False]
# fmt: on
EXPECTED_SCORE = 0.9384885764499121


def test_ts_selector(classifier, dataset, random_state):
    assert_selector_result(
        TSSelector(classifier, random_state=random_state),
        classifier,
        dataset,
        EXPECTED_SUPPORT,
        EXPECTED_SCORE,
    )
