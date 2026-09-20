import gc
import itertools
import sys
import threading
import weakref
from dataclasses import FrozenInstanceError

import pytest
import torch
from rcswx import CancellationToken, Limits, TensorSpec, _core, distance, edit_path, prepare
from rcswx.errors import BudgetExceeded, Cancelled, InputContractMismatch, OutputContractMismatch
from torch import nn

from tests.reference.ordered import optimum, projection

IDENTITY = (0, 0, False, 0.0, None, False)
RELU = (2, 0, False, 0.0, None, False)
LINEAR4 = (1, 4, True, 0.0, None, False)
LINEAR8 = (1, 8, True, 0.0, None, False)


def native_architecture(records, limits=None):
    limits = _core.Limits() if limits is None else limits
    return _core.Architecture(list(records), limits)


def witness(path):
    return tuple(
        (tag, -1 if source is None else source, -1 if target is None else target)
        for tag, source, target, _, _ in path.steps
    )


def test_complete_tiny_domain_matches_independent_unmerged_histories():
    alphabet = (IDENTITY, RELU, LINEAR4, LINEAR8)
    sequences = [
        sequence for length in (1, 2, 3) for sequence in itertools.product(alphabet, repeat=length)
    ]
    native = {sequence: native_architecture(sequence) for sequence in sequences}
    limits, token = _core.Limits(), CancellationToken()
    for left, right in itertools.product(sequences, repeat=2):
        expected_cost, expected_witness = optimum(left, right)
        path = _core.align(native[left], native[right], limits, token)
        measured, _ = _core.distance(native[left], native[right], limits, token)
        assert (path.cost_ticks, witness(path)) == (expected_cost, expected_witness), (left, right)
        assert measured == expected_cost
        for mask in range(1 << path.edit_count):
            expected, origins, cost = projection(left, right, expected_witness, mask)
            actual_origins = tuple(path.project(mask, limits, token))
            actual = tuple((left if side == 0 else right)[index] for side, index in actual_origins)
            assert (actual, actual_origins, path.selected_cost(mask)) == (expected, origins, cost)


def test_logical_width_and_bias_costs_do_not_include_realized_input_width():
    source = nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 4))
    target = nn.Sequential(nn.Linear(4, 16), nn.Linear(16, 4))
    a, b = (prepare(parent, input_spec=TensorSpec((2, 4))) for parent in (source, target))
    path = edit_path(a, b)
    assert path.cost_ticks == 1
    assert distance(a, b) == 0.25
    assert tuple((edit.kind, edit.cost_ticks) for edit in path.edits) == (("substitute", 1),)
    assert path.steps[-1][0] == 0
    without_bias = prepare(
        nn.Sequential(nn.Linear(4, 8, bias=False), nn.Linear(8, 4)), input_spec=TensorSpec((2, 4))
    )
    assert distance(a, without_bias) == 0.25


def test_zero_distance_across_contracts_cannot_be_an_executable_path():
    a = prepare(nn.Linear(4, 8), input_spec=TensorSpec((2, 4)))
    b = prepare(nn.Linear(16, 8), input_spec=TensorSpec((2, 16)))
    assert distance(a, b) == 0
    with pytest.raises(InputContractMismatch):
        edit_path(a, b)
    c = prepare(nn.Linear(4, 16), input_spec=TensorSpec((2, 4)))
    assert distance(a, c) == 0.25
    with pytest.raises(OutputContractMismatch):
        edit_path(a, c)
    with pytest.raises(TypeError):
        distance(a, nn.Linear(4, 8), input_spec=TensorSpec((2, 4)))


class Named(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.add_module(name, nn.Identity())
        self.name = name

    def forward(self, x):
        return self.get_submodule(self.name)(x)


def test_complete_witness_ignores_wrapper_names_and_retains_zero_matches():
    first = edit_path(
        Named("before"), nn.Sequential(nn.Identity(), nn.Identity()), input_spec=TensorSpec((2, 4))
    )
    renamed = edit_path(
        Named("after"), nn.Sequential(nn.Identity(), nn.Identity()), input_spec=TensorSpec((2, 4))
    )
    assert first.steps == renamed.steps == ((0, 0, 0, 0, None), (3, None, 1, 4, 0))
    assert first._project(0)[1] == ((0, 0),)
    assert first._project(1)[1] == ((0, 0), (1, 1))
    with pytest.raises(FrozenInstanceError):
        first.source = renamed.target


def test_paths_do_not_keep_live_modules_or_examples_alive():
    left, right, example = nn.Linear(4, 4), nn.Linear(4, 4), torch.ones(2, 4)
    references = (
        weakref.ref(left),
        weakref.ref(right),
        weakref.ref(left.weight),
        weakref.ref(example),
    )
    path = edit_path(left, right, example_inputs=(example,))
    del left, right, example
    gc.collect()
    assert all(reference() is None for reference in references)
    assert path._project(0)[1] == ((0, 0),)


def test_budget_and_cancellation_are_not_no_alignment_results():
    a = prepare(nn.Sequential(nn.Identity(), nn.ReLU()), input_spec=TensorSpec((2, 4)))
    with pytest.raises(BudgetExceeded):
        distance(a, a, limits=Limits(max_states=8))
    with pytest.raises(BudgetExceeded):
        edit_path(a, a, limits=Limits(max_predecessors=1))
    token = CancellationToken()
    token.cancel()
    with pytest.raises(Cancelled):
        distance(a, a, cancel=token)


def test_detached_kernel_allows_another_python_thread_to_cancel_it():
    limits = _core.Limits(max_states=4_000_000)
    source = native_architecture([IDENTITY, RELU] * 900, limits)
    target = native_architecture([RELU, IDENTITY] * 900, limits)
    token, gate, ready = CancellationToken(), threading.Event(), threading.Event()

    def canceller():
        ready.set()
        gate.wait()
        token.cancel()

    thread = threading.Thread(target=canceller)
    thread.start()
    ready.wait()
    old_interval = sys.getswitchinterval()
    try:
        # From gate.set through the native entry, only detach can hand the GIL
        # to the waiting Python thread. This is not a timing/sleep-based probe.
        sys.setswitchinterval(1000)
        gate.set()
        with pytest.raises(_core.CoreError) as failure:
            _core.distance(source, target, limits, token)
        assert failure.value.args[0] == "cancelled"
    finally:
        sys.setswitchinterval(old_interval)
        gate.set()
        thread.join(timeout=5)
    assert not thread.is_alive()
