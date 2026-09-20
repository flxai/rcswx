import gc
import json
import random
import weakref
from collections import OrderedDict

import pytest
import torch
from rcswx import (
    BudgetExceeded,
    Limits,
    MissingDonor,
    SamplingExhausted,
    StaleEditPath,
    TensorSpec,
    crossover,
    crossover_with_report,
    edit_path,
    prepare,
)
from torch import nn


def rejection_pair():
    return tuple(
        prepare(model, input_spec=TensorSpec((2, 4)))
        for model in (nn.Identity(), nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 4)))
    )


def test_shape_rejections_are_iid_with_replacement_and_do_not_realign():
    a, b = rejection_pair()
    path = edit_path(a, b)
    seed = (1 << 64) - 1
    with pytest.raises(SamplingExhausted) as failure:
        crossover(a, b, path=path, seed=seed, limits=Limits(max_attempts=3, max_states=1))
    report = failure.value.report
    assert report["attempts"] == 3
    assert [entry["mask"] for entry in report["rejections"]] == [1, 1, 1]
    assert report["alignment_computations"] == 0
    result = crossover_with_report(
        a, b, path=path, seed=seed, limits=Limits(max_attempts=4, max_states=1)
    )
    assert result.report["attempts"] == 4
    assert result.report["selected_mask"] == 2
    assert result.report["selected_cost_ticks"] == 4
    assert result.report["alignment_computations"] == 0
    assert tuple(result.model(torch.ones(2, 4)).shape) == (2, 4)
    fresh = crossover_with_report(a, b, seed=seed, limits=Limits(max_attempts=4))
    assert fresh.report["alignment_computations"] == 1
    assert fresh.report["selected_mask"] == result.report["selected_mask"]
    json.dumps(report, allow_nan=False)


def test_budget_failures_abort_instead_of_rejecting_a_proposal():
    a, b = rejection_pair()
    path = edit_path(a, b)
    with pytest.raises(BudgetExceeded) as failure:
        crossover(a, b, path=path, seed=42, limits=Limits(max_child_bytes=1))
    assert failure.value.resource == "max_child_bytes"
    with pytest.raises(BudgetExceeded) as failure:
        crossover(a, b, path=path, limits=Limits(max_edits=1))
    assert failure.value.resource == "max_edits"
    with pytest.raises(MissingDonor):
        crossover(a, b, path=path, inherit=True)


def test_zero_distance_source_values_modes_and_storage_are_independent():
    source, target = nn.Linear(4, 4).eval(), nn.Linear(4, 4).train()
    path = edit_path(source, target, input_spec=TensorSpec((2, 4)))
    with torch.no_grad():
        source.weight.fill_(3)
        source.bias.fill_(2)
        target.weight.fill_(7)
        target.bias.fill_(9)
    source.requires_grad_(False)
    result = crossover_with_report(source, target, path=path, inherit=True, seed=42)
    child = result.model
    assert result.report["selected_mask"] == 0
    assert result.report["selected_edit_ids"] == ()
    assert all(not module.training for module in child.modules())
    assert torch.equal(child(torch.ones(2, 4)), source(torch.ones(2, 4)))
    assert all(not parameter.requires_grad for parameter in child.parameters())
    assert all(entry["donor_side"] == 0 for entry in result.report["parameter_decisions"])
    donor_ptrs = {p.data_ptr() for model in (source, target) for p in model.parameters()}
    assert all(p.data_ptr() not in donor_ptrs for p in child.parameters())
    child[0].weight.fill_(11)
    assert torch.equal(source.weight, torch.full_like(source.weight, 3))


def test_selected_provenance_shape_fallback_and_current_trainability():
    source = nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 4)).eval()
    target = nn.Sequential(nn.Linear(4, 16), nn.Linear(16, 4))
    path = edit_path(source, target, input_spec=TensorSpec((2, 4)))
    with torch.no_grad():
        for parent, value in ((source, 3), (target, 7)):
            for parameter in parent.parameters():
                parameter.fill_(value)
                parameter.requires_grad_(False)
    result = crossover_with_report(source, target, path=path, inherit=True, init="zeros", seed=42)
    child = result.model
    assert result.report["selected_mask"] == 1
    assert torch.equal(child[0].weight, target[0].weight)
    assert child[1].weight.shape == (4, 16)
    assert torch.equal(child[1].weight, torch.zeros_like(child[1].weight))
    assert torch.equal(child[1].bias, source[1].bias)
    assert child[1].weight.requires_grad and not child[1].bias.requires_grad
    assert not child[0].weight.requires_grad and not child[0].bias.requires_grad
    old = [parameter.clone() for model in (source, target) for parameter in model.parameters()]
    child(torch.ones(2, 4)).sum().backward()
    torch.optim.SGD(child.parameters(), lr=0.01).step()
    assert all(
        torch.equal(before, after)
        for before, after in zip(
            old, (p for parent in (source, target) for p in parent.parameters()), strict=True
        )
    )
    assert all(p.grad is None for parent in (source, target) for p in parent.parameters())


def test_initialization_and_architecture_streams_preserve_caller_rng():
    a, b = rejection_pair()
    path = edit_path(a, b)
    torch_state, random_state = torch.get_rng_state(), random.getstate()
    zero = crossover_with_report(a, b, path=path, seed=42, init="zeros")
    fresh = crossover_with_report(a, b, path=path, seed=42, init="pytorch")
    repeat = crossover(a, b, path=path, seed=42, init="pytorch")
    assert zero.report["selected_mask"] == fresh.report["selected_mask"]
    assert all(torch.count_nonzero(p) == 0 for p in zero.model.parameters())
    assert all(
        torch.equal(a, b)
        for a, b in zip(fresh.model.parameters(), repeat.parameters(), strict=True)
    )
    assert any(
        not torch.equal(a, b)
        for a, b in zip(zero.model.parameters(), fresh.model.parameters(), strict=True)
    )
    assert torch.equal(torch.get_rng_state(), torch_state)
    assert random.getstate() == random_state


def test_stale_structure_orientation_and_bindings_never_realign():
    a, b = rejection_pair()
    path = edit_path(a, b)
    with pytest.raises(StaleEditPath):
        crossover(b, a, path=path)
    changed = prepare(nn.ReLU(), input_spec=TensorSpec((2, 4)))
    with pytest.raises(StaleEditPath):
        crossover(changed, b, path=path)
    renamed = prepare(
        nn.Sequential(OrderedDict([("one", nn.Linear(4, 8)), ("two", nn.Linear(8, 4))])),
        input_spec=TensorSpec((2, 4)),
    )
    assert renamed.architecture_key == b.architecture_key
    with pytest.raises(StaleEditPath):
        crossover(a, renamed, path=path)


def test_reports_do_not_retain_parents_and_children_reenter_later_generations():
    source = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 4))
    target = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 4))
    refs = weakref.ref(source), weakref.ref(target), weakref.ref(source[0].weight)
    path = edit_path(source, target, input_spec=TensorSpec((2, 4)))
    result = crossover_with_report(source, target, path=path, seed=42, inherit=True)
    report, child = result.report, result.model
    del source, target, result, path
    gc.collect()
    assert all(reference() is None for reference in refs)
    json.dumps(report, allow_nan=False)
    for seed in range(8):
        architecture = prepare(child, input_spec=TensorSpec((2, 4)))
        child = crossover(architecture, architecture, seed=seed)
        assert (
            prepare(child, input_spec=TensorSpec((2, 4))).architecture_key
            == architecture.architecture_key
        )
        child(torch.ones(2, 4)).sum().backward()
        assert all(p.grad is not None for p in child.parameters())
