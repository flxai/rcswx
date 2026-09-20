import subprocess
import sys

import pytest
import torch
from rcswx import (
    Limits,
    SamplingExhausted,
    TensorSpec,
    UnsupportedArchitecture,
    UnsupportedConfiguration,
    UnsupportedEffect,
    UnsupportedSharing,
    crossover,
    crossover_with_report,
    edit_path,
    prepare,
)
from rcswx.materialize import copied_state
from torch import nn
from torch.nn import functional as functional


@pytest.mark.parametrize(
    "affine,tracking", [(True, True), (False, True), (True, False), (False, False)]
)
def test_batch_norm_numerical_roundtrip_and_owned_state_in_both_modes(affine, tracking):
    source = nn.BatchNorm1d(4, affine=affine, track_running_stats=tracking, momentum=None).eval()
    with torch.no_grad():
        if affine:
            source.weight.fill_(2)
            source.bias.fill_(0.25)
        if tracking:
            source.running_mean.fill_(0.5)
            source.running_var.fill_(1.5)
            source.num_batches_tracked.fill_(9)
    spec = TensorSpec((3, 4))
    architecture = prepare(source, input_spec=spec)
    child = copied_state(architecture, source)
    for training in (False, True):
        source.train(training)
        child.train(training)
        x = torch.linspace(-1, 2, 12).reshape(3, 4)
        y = x.clone().requires_grad_()
        torch.testing.assert_close(child(y), source(x))
        child(y).square().sum().backward()
        assert y.grad is not None
        assert all(p.grad is None for p in source.parameters())
    donor_ptrs = {t.data_ptr() for t in (*source.parameters(), *source.buffers())}
    assert all(t.data_ptr() not in donor_ptrs for t in (*child.parameters(), *child.buffers()))
    assert prepare(child, input_spec=spec).architecture_key == architecture.architecture_key


def test_trained_after_path_affine_values_copy_but_statistics_reset():
    source, target = nn.BatchNorm1d(4), nn.BatchNorm1d(4)
    path = edit_path(source, target, input_spec=TensorSpec((3, 4)))
    source(torch.linspace(-1, 2, 12).reshape(3, 4)).square().sum().backward()
    torch.optim.SGD(source.parameters(), lr=0.02).step()
    source.eval().requires_grad_(False)
    before = tuple(t.clone() for t in (*source.parameters(), *source.buffers()))
    rng = torch.get_rng_state()
    result = crossover_with_report(source, target, path=path, inherit=True, seed=7)
    child = result.model[0]
    assert torch.equal(child.weight, source.weight) and torch.equal(child.bias, source.bias)
    assert not child.weight.requires_grad and not child.bias.requires_grad
    assert child.weight.grad is None and child.bias.grad is None
    assert torch.equal(child.running_mean, torch.zeros(4))
    assert torch.equal(child.running_var, torch.ones(4))
    assert child.num_batches_tracked.item() == 0
    assert child.num_batches_tracked.dtype is torch.int64
    assert all(
        torch.equal(a, b)
        for a, b in zip(before, (*source.parameters(), *source.buffers()), strict=True)
    )
    assert torch.equal(torch.get_rng_state(), rng)
    assert not child.training
    child.train()(torch.ones(3, 4))
    assert child.num_batches_tracked.item() == 1
    assert source.num_batches_tracked.item() == 1
    assert all(
        torch.equal(a, b)
        for a, b in zip(before, (*source.parameters(), *source.buffers()), strict=True)
    )


def test_dependent_normalization_channels_initialize_whole_affine_roles():
    source = nn.Sequential(nn.Linear(4, 8), nn.BatchNorm1d(8), nn.Linear(8, 4))
    target = nn.Sequential(nn.Linear(4, 16), nn.BatchNorm1d(16), nn.Linear(16, 4))
    path = edit_path(source, target, input_spec=TensorSpec((2, 4)))
    with torch.no_grad():
        source[1].weight.fill_(3)
        target[1].weight.fill_(7)
    rng = torch.get_rng_state()
    result = crossover_with_report(source, target, path=path, inherit=True, seed=42)
    child = result.model
    assert path.cost_ticks == 1
    assert result.report["selected_mask"] == 1
    assert child[1].num_features == 16
    assert torch.equal(child[1].weight, torch.ones(16))
    assert torch.equal(child[1].bias, torch.zeros(16))
    assert child[1].weight.requires_grad and child[1].bias.requires_grad
    assert torch.equal(torch.get_rng_state(), rng)
    for training in (False, True):
        child.train(training)
        child(torch.ones(2, 4)).sum().backward()
    assert (
        prepare(child, input_spec=TensorSpec((2, 4))).architecture_key
        == path.target.architecture_key
    )
    zero = crossover(source, target, path=path, inherit=False, init="zeros", seed=42)
    assert torch.equal(zero[1].weight, torch.zeros(16))
    assert torch.equal(zero[1].running_var, torch.ones(16))


def test_eval_only_batch_norm_and_recombined_training_invalidity_reject():
    with pytest.raises(UnsupportedConfiguration):
        prepare(nn.BatchNorm1d(4).eval(), input_spec=TensorSpec((1, 4)))
    with pytest.raises(UnsupportedConfiguration):
        prepare(nn.BatchNorm1d(4).eval(), input_spec=TensorSpec((1, 4, 1)))
    source = nn.Sequential(nn.Linear(1, 2), nn.BatchNorm1d(2), nn.Linear(2, 1)).eval()
    target = nn.Sequential(nn.Linear(1, 1), nn.ReLU(), nn.Linear(1, 1))
    path = edit_path(source, target, input_spec=TensorSpec((1, 2, 1)))
    with pytest.raises(SamplingExhausted) as failure:
        crossover(source, target, path=path, seed=(1 << 64) - 1, limits=Limits(max_attempts=3))
    assert [item["reason"] for item in failure.value.report["rejections"]] == [
        "operator_configuration"
    ] * 3
    result = crossover_with_report(
        source, target, path=path, seed=(1 << 64) - 1, limits=Limits(max_attempts=4)
    )
    assert result.report["attempts"] == 4 and result.report["selected_mask"] == 2
    result.model.train()(torch.ones(1, 2, 1)).sum().backward()


def test_registered_state_does_not_admit_unknown_buffers_or_aliases():
    source = nn.BatchNorm1d(4)
    source.running_mean.requires_grad_(True)
    with pytest.raises(UnsupportedConfiguration):
        prepare(source, input_spec=TensorSpec((2, 4)))
    source = nn.BatchNorm1d(4)
    source.register_buffer("extra", torch.zeros(1))
    with pytest.raises(UnsupportedConfiguration):
        prepare(source, input_spec=TensorSpec((2, 4)))
    source = nn.BatchNorm1d(4)
    source.running_var = source.running_mean
    with pytest.raises(UnsupportedSharing):
        prepare(source, input_spec=TensorSpec((2, 4)))


def test_dropout_conformance_owns_stochastic_execution_in_a_subprocess():
    script = """
import torch
from torch import nn
from rcswx import TensorSpec, prepare, crossover
for p in (0.0, 0.5, 1.0):
    source = nn.Dropout(p)
    state = torch.get_rng_state()
    architecture = prepare(source, input_spec=TensorSpec((2, 8)))
    child = crossover(architecture, architecture, seed=42)
    assert torch.equal(state, torch.get_rng_state())
    for training in (False, True):
        source.train(training)
        child.train(training)
        x = torch.ones(2, 8, requires_grad=True)
        y = torch.ones(2, 8, requires_grad=True)
        torch.manual_seed(987)
        left = source(x)
        torch.manual_seed(987)
        right = child(y)
        assert torch.equal(left, right)
        left.sum().backward()
        right.sum().backward()
        assert torch.equal(x.grad, y.grad)
        assert prepare(child, input_spec=TensorSpec((2, 8))).architecture_key == architecture.architecture_key
print("dropout train/eval and gradient conformance passed")
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr


def test_functional_dropout_and_inplace_specializations_stay_unsupported():
    class Functional(nn.Module):
        def forward(self, x):
            return functional.dropout(x, p=0.0, training=self.training)

    with pytest.raises(UnsupportedArchitecture):
        prepare(Functional(), input_spec=TensorSpec((2, 4)))
    with pytest.raises(UnsupportedEffect):
        prepare(nn.Dropout(0.0, inplace=True), input_spec=TensorSpec((2, 4)))
