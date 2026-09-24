import sys

import pytest
from rcswx import _core
from rcswx.recursive import MatrixOperation
from rcswx.sampling import probabilities, select_operations


def operations(costs):
    return [
        MatrixOperation(op_id=index, value=cost, enabler_ops=[], disabler_ops=[])
        for index, cost in enumerate(costs)
    ]


def test_native_default_needs_no_reference_module(monkeypatch):
    monkeypatch.delitem(sys.modules, "rcswx.sampling_reference", raising=False)
    selected = select_operations(operations([0.25, 0.25]), seed=19)
    assert all(operation.id in {0, 1} for operation in selected)
    assert "rcswx.sampling_reference" not in sys.modules


def test_native_sampling_does_not_advance_numpy_global_stream():
    np = pytest.importorskip("numpy")
    np.random.seed(31)
    before = np.random.get_state()
    select_operations(operations([0.25, 0.25]), seed=31)
    after = np.random.get_state()
    assert before[0] == after[0]
    assert before[1].tobytes() == after[1].tobytes()
    assert before[2:] == after[2:]


def test_native_probability_vectors_match_explicit_reference_policy():
    np = pytest.importorskip("numpy")
    pytest.importorskip("scipy")
    from rcswx.sampling_reference import probabilities as reference_probabilities

    for values in ([0.25], [0.25, 0.25, 0.25, 0.25], [0.25, 0.5, 0.75]):
        for skewness in (-100.0, -3.0, 0.0, 2.0, 100.0):
            native = probabilities(values, skewness)
            reference = reference_probabilities(values, skewness)
            np.testing.assert_allclose(native, reference, atol=1e-12, rtol=1e-10)
            assert 0.5 * sum(abs(left - right) for left, right in zip(native, reference)) < 1e-9


def test_native_keeps_fractional_and_final_mask_index_failures():
    with pytest.raises(IndexError):
        probabilities([0.125], 0)
    # The final enumerated value controls L. Replacing it with max(values) would
    # make this input succeed with a two-entry grid.
    with pytest.raises(IndexError):
        probabilities([0.5, 0.25], 0)


def test_native_fractional_indices_use_reference_negative_indexing():
    np = pytest.importorskip("numpy")
    pytest.importorskip("scipy")
    from rcswx.sampling_reference import probabilities as reference_probabilities

    values = [0.125, 0.25]
    native = probabilities(values, -2.0)
    reference = reference_probabilities(values, -2.0)
    np.testing.assert_allclose(native, reference, atol=1e-12, rtol=1e-10)


def test_native_rng_seed_encoding_high53_uniform_and_state_restoration():
    seed = (1 << 255) + 0x1234
    from_integer = _core.NativeRng(seed)
    from_bytes = _core.NativeRng(seed.to_bytes(32, "little"))
    assert [from_integer.next_u64() for _ in range(3)] == [from_bytes.next_u64() for _ in range(3)]

    words = _core.NativeRng(7)
    word = words.next_u64()
    uniform = _core.NativeRng(7)
    assert uniform.uniform() == (word >> 11) / 2**53

    state = uniform.state()
    restored = _core.NativeRng.from_state(state)
    assert [uniform.next_u64() for _ in range(4)] == [restored.next_u64() for _ in range(4)]
    assert state["version"] == 1
    assert state["algorithm"] == "ChaCha12"
    assert all(isinstance(state[name], str) for name in ("seed", "stream", "word_position"))


def test_native_rng_enforces_exact_seed_range_and_byte_length():
    with pytest.raises(OverflowError):
        _core.NativeRng(-1)
    with pytest.raises(OverflowError):
        _core.NativeRng(1 << 256)
    with pytest.raises(ValueError):
        _core.NativeRng(b"\0" * 31)


def test_invalid_native_selection_does_not_advance_supplied_rng():
    rng = _core.NativeRng(42)
    before = rng.state()
    with pytest.raises(IndexError):
        select_operations(operations([0.125]), rng=rng)
    assert rng.state() == before


def test_reference_mode_rejects_native_rng_arguments_before_optional_import(monkeypatch):
    monkeypatch.delitem(sys.modules, "rcswx.sampling_reference", raising=False)
    with pytest.raises(ValueError, match="only by sampler='native'"):
        select_operations([], sampler="reference", seed=1)
    assert "rcswx.sampling_reference" not in sys.modules


def test_native_categorical_distribution_smoke():
    records = [(0, 0.25, [], [])]
    selected = 0
    samples = 2048
    for seed in range(samples):
        mask, _cost = _core.native_select(records, 0.0, _core.NativeRng(seed))
        selected += mask == "1"
    assert abs(selected - samples / 2) < 160


def budgeted_choice_plan(count, **limits):
    from rcswx import Alignment, Architecture

    def tree(name, width):
        if width == 1:
            return ("computation", (name,))
        middle = width // 2
        return ("sequential", tree(name, middle), tree(name, width - middle))

    return Alignment(
        Architecture.from_tree(tree("relu", count)),
        Architecture.from_tree(tree("sigmoid", count)),
        limits=limits or None,
    )._native


def test_budgeted_enumeration_batches_host_checks_without_changing_choices():
    plan = budgeted_choice_plan(12)
    expected = plan.combinations()
    calls = 0

    def check():
        nonlocal calls
        calls += 1
        return True

    actual = plan.combinations(check)
    assert actual == expected
    assert len(actual[0]) == 4096
    # Regression guard: an RSS query must not accompany every mask/tree visit.
    assert 1 < calls <= 64


def test_budgeted_enumeration_rejects_initial_host_memory_failure():
    with pytest.raises(MemoryError):
        budgeted_choice_plan(1).combinations(lambda: False)


@pytest.mark.parametrize("count", [1, 12])
def test_budgeted_enumeration_rechecks_memory_before_success(count):
    plan = budgeted_choice_plan(count)
    calls = 0

    def check():
        nonlocal calls
        calls += 1
        return calls < 2

    # A short operation needs its final poll; a longer one needs periodic polls.
    with pytest.raises(MemoryError):
        plan.combinations(check)


def test_budgeted_enumeration_preserves_host_exception():
    # Without periodic polls, enumeration would hit its output quota instead.
    plan = budgeted_choice_plan(12, max_output=1024)
    failure = RuntimeError("host cancelled enumeration")
    calls = 0

    def check():
        nonlocal calls
        calls += 1
        if calls == 2:
            raise failure
        return True

    with pytest.raises(RuntimeError) as raised:
        plan.combinations(check)
    assert raised.value is failure


def test_batched_host_polling_does_not_disable_native_output_limit():
    plan = budgeted_choice_plan(12, max_output=1024)
    with pytest.raises(MemoryError, match="output limit"):
        plan.combinations(lambda: True)
