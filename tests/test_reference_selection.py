import numpy as np
import pytest
from rcswx.recursive import MatrixOperation
from rcswx.sampling import select_operations, valid_combinations

from tests.reference.original import load


def operations(costs):
    return [
        MatrixOperation(op_id=index, value=np.float64(cost), enabler_ops=[], disabler_ops=[])
        for index, cost in enumerate(costs)
    ]


def cases():
    independent = operations([1, 2])
    last_not_maximum = operations([1, 3])
    last_not_maximum[0].disabler_ops = [last_not_maximum[1]]
    grouped = operations([0.125, 0.25, 1, 2])
    grouped[0].enabler_ops = [[grouped[1]], grouped[3]]
    grouped[0].disabler_ops = [[grouped[2]], grouped[1], grouped[2]]
    grouped[2].enabler_ops = [grouped[0]]
    grouped[2].disabler_ops = [grouped[1]]
    invalid_all = operations([1, 1])
    invalid_all[0].disabler_ops = [invalid_all[1]]
    return [
        independent,
        grouped,
        invalid_all,
        last_not_maximum,
        operations([0.125]),
        operations([0]),
        operations([]),
        operations([float("nan")]),
        operations([float("inf")]),
    ]


def state():
    value = np.random.get_state()
    return value[0], value[1].tobytes(), value[2], value[3], value[4]


@pytest.mark.parametrize("edits", cases())
def test_valid_mask_order_costs_and_reference_dependency_rules(edits):
    expected = load().algorithm.combinations(edits)
    masks, costs = valid_combinations(edits)
    assert masks == list(expected)
    assert costs == list(expected.values())


@pytest.mark.parametrize("edits", cases())
@pytest.mark.parametrize("skewness", [-3, 0, 2])
def test_selector_outcome_and_rng_transition_including_failures(edits, skewness):
    for seed in range(64):
        np.random.seed(seed)
        before = state()
        try:
            expected = load().algorithm.select_operations(edits, skewness)
        except Exception as error:
            expected_error = type(error)
            after = state()
            np.random.seed(seed)
            with pytest.raises(expected_error):
                select_operations(edits, skewness)
            assert state() == after == before
        else:
            after = state()
            np.random.seed(seed)
            actual = select_operations(edits, skewness)
            assert [op.id for op in actual] == [op.id for op in expected]
            assert state() == after


def test_dependency_pruning_has_no_machine_word_mask_limit():
    edits = operations([1] * 70)
    # Mutual exclusion leaves exactly N+1 valid subsets without enumerating 2**N.
    for index, edit in enumerate(edits):
        edit.enabler_ops = [[] for _ in range(index)]
        edit.disabler_ops = [[other] for other in edits[:index]]
    masks, costs = valid_combinations(edits)
    assert masks == ["0" * 70] + ["0" * (69 - index) + "1" + "0" * index for index in range(70)]
    assert costs == [0] + [1] * 70
