from metawrappers import GASelector
from tests.selectors.conftest import assert_selector_result


# fmt: off
EXPECTED_SUPPORT = [False, False, False, False, False, False, False, False, False, False, False,
                    False, False, False, False, False, False, False, False, True, False, True, True,
                    False, False, False, False, False, False, False]
# fmt: on
EXPECTED_SCORE = 0.9420035149384886


def test_ga_selector(classifier, dataset, random_state):
    assert_selector_result(
        GASelector(classifier, random_state=random_state),
        classifier,
        dataset,
        EXPECTED_SUPPORT,
        EXPECTED_SCORE,
    )
