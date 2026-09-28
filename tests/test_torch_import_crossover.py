"""Strict offspring validation without changing portable selection semantics."""

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest
import torch
from rcswx import crossover_with_report as portable_crossover
from rcswx.torch import (
    UnsupportedModuleError,
    build,
    capture,
    crossover_imported,
    import_model,
    load,
    save,
)
from torch import nn


@pytest.fixture(autouse=True)
def isolated_randomness():
    numpy_state = np.random.get_state()
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(31)
            yield
    finally:
        np.random.set_state(numpy_state)


def parents(x):
    first = nn.Sequential(nn.Linear(8, 16), nn.ReLU())
    second = nn.Sequential(nn.Linear(8, 32), nn.Softmax(dim=-1))
    return tuple(import_model(model, example_inputs=(x,)) for model in (first, second))


@pytest.mark.parametrize("sampler", ["native", "reference"])
def test_validation_preserves_portable_selection_and_initialization_rng(sampler):
    x = torch.randn(4, 8)
    first, second = parents(x)
    options = {"seed": 0} if sampler == "native" else {}
    np.random.seed(0)
    portable = portable_crossover(
        first.architecture, second.architecture, sampler=sampler, **options
    )
    torch.manual_seed(42)
    expected = build(portable.child)
    expected_rng = torch.random.get_rng_state()
    torch.manual_seed(42)
    np.random.seed(0)
    result = crossover_imported(first, second, validation_inputs=(x,), sampler=sampler, **options)
    assert result.architecture == portable.child
    torch.testing.assert_close(result.child(x), expected(x), rtol=0, atol=0)
    assert torch.equal(torch.random.get_rng_state(), expected_rng)


def test_mixed_offspring_trains_and_roundtrips_with_child_bindings(tmp_path):
    x = torch.randn(4, 8)
    first, second = parents(x)
    result = crossover_imported(first, second, validation_inputs=(x,), seed=0)
    assert result.architecture not in (first.architecture, second.architecture)
    optimizer = torch.optim.SGD(result.child.parameters(), lr=0.1)
    before = result.child(x).detach().clone()
    result.child(x).square().mean().backward()
    optimizer.step()
    result.child.eval()
    trained = result.child(x).detach()
    assert not torch.equal(before, trained)
    checkpoint = tmp_path / "trained.pt"
    save(result.child, checkpoint)
    restored = load(checkpoint)
    torch.testing.assert_close(restored(x), trained, rtol=0, atol=0)
    assert not restored.training
    assert capture(restored).architecture == result.architecture


def test_runtime_gate_does_not_inherit_or_update_parent_and_child_buffers():
    x = torch.randn(4, 16, 7)
    source = nn.Sequential(nn.BatchNorm1d(16), nn.ReLU())
    with torch.no_grad():
        source[0].running_mean.fill_(7)
        source[0].num_batches_tracked.fill_(12)
        source[0].weight.fill_(3)
    state = deepcopy(source.state_dict())
    imported = import_model(source, example_inputs=(x,))
    result = crossover_imported(imported, imported, validation_inputs=(x,), seed=0)
    norm = next(module for module in result.child.modules() if isinstance(module, nn.BatchNorm1d))
    torch.testing.assert_close(norm.running_mean, torch.zeros_like(norm.running_mean))
    torch.testing.assert_close(norm.weight, torch.ones_like(norm.weight))
    assert norm.num_batches_tracked.item() == 0
    for key, value in state.items():
        torch.testing.assert_close(source.state_dict()[key], value, rtol=0, atol=0)


def test_invalid_sample_is_rejected_before_initialization_instead_of_resampled():
    class Residual(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = nn.Linear(16, 16)

        def forward(self, x):
            return self.layer(x).relu() + x

    x = torch.randn(4, 16)
    first = import_model(Residual(), example_inputs=(x,))
    second = import_model(
        nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 16)), example_inputs=(x,)
    )
    # This real native sample joins a width-32 branch to a width-16 identity.
    rng = torch.random.get_rng_state()
    with pytest.raises(UnsupportedModuleError):
        crossover_imported(first, second, validation_inputs=(x,), seed=5)
    assert torch.equal(torch.random.get_rng_state(), rng)


def test_changed_input_signatures_require_explicit_materialization_override():
    x = torch.randn(4, 8)
    first, second = parents(x)
    with pytest.raises(UnsupportedModuleError):
        crossover_imported(first, second, validation_inputs=(x.double(),), seed=0)
    result = crossover_imported(
        first,
        second,
        validation_inputs=(x.double(),),
        seed=0,
        build_options={"dtype": torch.float64, "training": False},
    )
    assert result.child(x.double()).dtype == torch.float64
    assert not result.child.training
    with pytest.raises(UnsupportedModuleError):
        crossover_imported(first, second, validation_inputs=(x[:2],), seed=0)


def test_unverified_and_inconsistent_snapshots_are_rejected():
    x = torch.randn(4, 8)
    first, second = parents(x)
    unverified = capture(build(first))
    inconsistent = replace(first, input_spec={**first.input_spec, "shape": [4, 9]})
    for parent in (unverified, inconsistent):
        with pytest.raises(UnsupportedModuleError):
            crossover_imported(parent, second, validation_inputs=(x,), seed=0)
