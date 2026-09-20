import json
import random

import pytest
import torch
from rcswx import TensorSpec, prepare
from rcswx.errors import BudgetExceeded
from rcswx.materialize import copied_state, materialize
from rcswx.operators import infer
from rcswx.policies import Limits
from rcswx.types import Operation
from torch import nn


class IndependentlyAuthored(nn.Module):
    def __init__(self):
        super().__init__()
        self.finish = nn.Linear(7, 3)
        self.activate = nn.ReLU()
        self.begin = nn.Linear(4, 7)

    def forward(self, x):
        hidden = self.begin(x)
        activated = self.activate(hidden)
        return self.finish(activated)


@pytest.mark.parametrize(
    "factory", [lambda: nn.Identity(), lambda: nn.Linear(4, 3), IndependentlyAuthored]
)
def test_copied_state_preserves_outputs_input_and_parameter_gradients(factory):
    parent = factory()
    architecture = prepare(parent, input_spec=TensorSpec((2, 4)))
    child = copied_state(architecture, parent)
    first = torch.linspace(-2, 2, 8).reshape(2, 4).requires_grad_()
    second = first.detach().clone().requires_grad_()
    left, right = parent(first), child(second)
    torch.testing.assert_close(left, right)
    left.square().sum().backward()
    right.square().sum().backward()
    torch.testing.assert_close(first.grad, second.grad)
    for index, binding in enumerate(architecture.bindings):
        donor = (
            parent.get_submodule(binding.module_path) if binding.module_path is not None else None
        )
        for parameter in binding.parameters:
            expected = donor._parameters[parameter.name]
            actual = child[index]._parameters[parameter.name]
            torch.testing.assert_close(expected.grad, actual.grad)
            assert expected.untyped_storage().data_ptr() != actual.untyped_storage().data_ptr()
    assert (
        prepare(child, input_spec=architecture.input_spec).architecture_key
        == architecture.architecture_key
    )


def build(architecture, *, seed=123, init="pytorch"):
    inference = infer(architecture.operations, architecture.input_spec, limits=Limits())
    return materialize(
        architecture.operations,
        architecture.input_spec,
        inference=inference,
        training=architecture.training,
        init=init,
        initialization_seed=seed,
        inherit=False,
        donors=None,
        donor_architectures=None,
        provenance=tuple((0, i) for i in range(len(architecture.operations))),
        limits=Limits(),
    )[0]


def test_private_initialization_preserves_caller_rng_and_ignores_frozen_donor_flags():
    parent = IndependentlyAuthored().eval().requires_grad_(False)
    architecture = prepare(parent, input_spec=TensorSpec((2, 4)))
    torch_state, python_state = torch.get_rng_state().clone(), random.getstate()
    first = build(architecture)
    assert torch.equal(torch_state, torch.get_rng_state())
    assert python_state == random.getstate()
    assert all(not module.training for module in first.modules())
    assert all(parameter.requires_grad for parameter in first.parameters())
    torch.rand(9)
    second = build(architecture)
    for left, right in zip(first.parameters(), second.parameters(), strict=True):
        assert torch.equal(left, right)
        assert left.untyped_storage().data_ptr() != right.untyped_storage().data_ptr()
    zero = build(architecture, init="zeros")
    assert all(torch.count_nonzero(parameter).item() == 0 for parameter in zero.parameters())


def test_dependent_width_initializes_whole_weight_but_copies_chosen_bias():
    source = nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 4))
    target = nn.Sequential(nn.Linear(4, 16), nn.Linear(16, 4))
    with torch.no_grad():
        source[1].weight.fill_(3)
        source[1].bias.fill_(2)
        target[0].weight.fill_(1)
        target[1].weight.fill_(7)
        target[1].bias.fill_(9)
    source[1].bias.requires_grad_(False)
    a, b = (prepare(parent, input_spec=TensorSpec((2, 4))) for parent in (source, target))
    operations, provenance = (b.operations[0], a.operations[1]), ((1, 0), (0, 1))
    inference = infer(operations, a.input_spec, limits=Limits())
    before = tuple(
        parameter.detach().clone()
        for parent in (source, target)
        for parameter in parent.parameters()
    )
    child, decisions = materialize(
        operations,
        a.input_spec,
        inference=inference,
        training=True,
        init="zeros",
        initialization_seed=1,
        inherit=True,
        donors=(source, target),
        donor_architectures=(a, b),
        provenance=provenance,
        limits=Limits(),
    )
    assert child[1].in_features == 16
    assert torch.count_nonzero(child[1].weight).item() == 0
    assert torch.equal(child[1].bias, source[1].bias)
    assert child[1].weight.requires_grad and not child[1].bias.requires_grad
    assert torch.equal(child[0].weight, target[0].weight)
    assert json.loads(json.dumps(decisions))[2]["action"] == "initialized"
    donor_storage = {
        p.untyped_storage().data_ptr() for parent in (source, target) for p in parent.parameters()
    }
    assert all(
        p.untyped_storage().data_ptr() not in donor_storage and p.grad is None and p.grad_fn is None
        for p in child.parameters()
    )
    optimizer = torch.optim.SGD(child.parameters(), lr=0.1)
    child(torch.ones(2, 4)).sum().backward()
    optimizer.step()
    for saved, current in zip(
        before, (p for parent in (source, target) for p in parent.parameters()), strict=True
    ):
        assert torch.equal(saved, current)
        assert current.grad is None


def test_dimensions_and_child_bytes_are_checked_before_tensor_allocation():
    operations = (Operation("linear", (("bias", True), ("out_features", 1000))),)
    with pytest.raises(BudgetExceeded):
        infer(operations, TensorSpec((2, 1000)), limits=Limits(max_child_bytes=1024))
    with pytest.raises(BudgetExceeded):
        TensorSpec((1 << 40, 1 << 40))


def test_construction_under_inference_mode_still_returns_trainable_parameters():
    architecture = prepare(nn.Linear(4, 3), input_spec=TensorSpec((2, 4)))
    with torch.inference_mode():
        child = build(architecture)
        assert torch.is_inference_mode_enabled()
    assert not child[0].weight.is_inference()
    example = torch.ones(2, 4, requires_grad=True)
    child(example).sum().backward()
    assert torch.equal(child[0].weight.grad, torch.full((3, 4), 2.0))
