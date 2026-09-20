import gc
import weakref

import pytest
import torch
from rcswx import TensorSpec, prepare
from rcswx.errors import (
    BudgetExceeded,
    UnsupportedArchitecture,
    UnsupportedConfiguration,
    UnsupportedEffect,
    UnsupportedSharing,
)
from rcswx.policies import Limits
from torch import nn


class ReverseRegistration(nn.Module):
    def __init__(self):
        super().__init__()
        self.output = nn.Linear(5, 3)
        self.activation = nn.ReLU()
        self.input = nn.Linear(4, 5)

    def forward(self, x):
        return self.output(self.activation(self.input(x)))


def test_capture_follows_dataflow_not_registration_order():
    parent = ReverseRegistration()
    architecture = prepare(parent, input_spec=TensorSpec((2, 4)))
    assert tuple(binding.module_path for binding in architecture.bindings) == (
        "input",
        "activation",
        "output",
    )
    assert architecture.output_spec.shape == (2, 3)
    assert architecture.operations[0].attributes == (("bias", True), ("out_features", 5))
    assert architecture.bindings[0].parameters[0].shape == (5, 4)


def test_normalization_retains_real_identities_and_drops_empty_scaffolding():
    nested = nn.Sequential(
        nn.Sequential(), nn.Identity(), nn.Sequential(nn.ReLU(), nn.Sequential()), nn.Identity()
    )
    flat = nn.Sequential(nn.Identity(), nn.ReLU(), nn.Identity())
    contract = TensorSpec((2, 4))
    prepared = prepare(nested, input_spec=contract)
    assert prepared.architecture_key == prepare(flat, input_spec=contract).architecture_key
    assert tuple(op.kind for op in prepared.operations) == ("identity", "relu", "identity")
    assert prepared.architecture_key != prepare(nn.ReLU(), input_spec=contract).architecture_key
    assert (
        prepare(nn.Sequential(), input_spec=contract).architecture_key
        == prepare(nn.Identity(), input_spec=contract).architecture_key
    )


def test_capture_rejects_inconsistent_realized_linear_input():
    parent = nn.Linear(4, 3)
    parent.in_features = 5
    with pytest.raises(UnsupportedConfiguration):
        prepare(parent, input_spec=TensorSpec((2, 4)))


def test_capture_keys_ignore_names_training_weights_and_trainability():
    parent = ReverseRegistration()
    first = prepare(parent, input_spec=TensorSpec((2, 4)))
    parent.eval().requires_grad_(False)
    with torch.no_grad():
        parent.input.weight.add_(10)
    second = prepare(parent, input_spec=TensorSpec((2, 4)))
    renamed = nn.Sequential(nn.Linear(4, 5), nn.ReLU(), nn.Linear(5, 3)).eval()
    third = prepare(renamed, input_spec=TensorSpec((2, 4)))
    assert first.canonical == second.canonical == third.canonical
    assert first.architecture_key == second.architecture_key == third.architecture_key
    assert first.binding_key == second.binding_key != third.binding_key
    assert first.training and not second.training


def test_shared_modules_and_disjoint_storage_views_are_rejected():
    shared = nn.Linear(4, 4)
    with pytest.raises(UnsupportedSharing):
        prepare(nn.Sequential(shared, shared), input_spec=TensorSpec((2, 4)))
    backing = torch.zeros(32)
    first, second = nn.Linear(4, 4), nn.Linear(4, 4)
    first.weight = nn.Parameter(backing[:16].view(4, 4))
    second.weight = nn.Parameter(backing[16:].view(4, 4))
    with pytest.raises(UnsupportedSharing):
        prepare(nn.Sequential(first, second), input_spec=TensorSpec((2, 4)))


def test_hooks_are_rejected_without_execution():
    parent, calls = nn.Linear(4, 4), []
    handle = parent.register_forward_pre_hook(lambda *_: calls.append(True))
    try:
        with pytest.raises(UnsupportedEffect):
            prepare(parent, input_spec=TensorSpec((2, 4)))
        assert calls == []
    finally:
        handle.remove()


class ModeBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.left, self.right = nn.Linear(4, 4), nn.Linear(4, 4)

    def forward(self, x):
        return self.left(x) if self.training else self.right(x)


def test_two_mode_capture_does_not_freeze_wrapper_control_flow_or_mutate_modes():
    parent = ModeBranch().eval()
    before = torch.get_rng_state().clone()
    with pytest.raises(UnsupportedEffect):
        prepare(parent, input_spec=TensorSpec((2, 4)))
    assert all(not module.training for module in parent.modules())
    assert torch.equal(before, torch.get_rng_state())
    parent.left.train()
    with pytest.raises(UnsupportedConfiguration):
        prepare(parent, input_spec=TensorSpec((2, 4)))


class Residual(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(4, 4)

    def forward(self, x):
        return x + self.layer(x)


def test_structured_graph_is_not_silently_flattened():
    with pytest.raises(UnsupportedArchitecture):
        prepare(Residual(), input_spec=TensorSpec((2, 4)))


def test_capture_rejects_inplace_and_noncontiguous_inputs():
    with pytest.raises(UnsupportedEffect):
        prepare(nn.ReLU(inplace=True), input_spec=TensorSpec((2, 4)))
    with pytest.raises(UnsupportedConfiguration):
        prepare(nn.Identity(), example_inputs=(torch.zeros(2, 4).t(),))
    with pytest.raises(ValueError):
        prepare(nn.Identity(), example_inputs=(torch.zeros(2, 4),), input_spec=TensorSpec((2, 4)))


def test_metadata_does_not_retain_model_parameters_or_example():
    parent, example = ReverseRegistration(), torch.zeros(2, 4)
    refs = weakref.ref(parent), weakref.ref(parent.input.weight), weakref.ref(example)
    architecture = prepare(parent, example_inputs=(example,))
    del parent, example
    gc.collect()
    assert all(reference() is None for reference in refs)
    assert architecture.output_spec == TensorSpec((2, 3))


def test_capture_depth_and_node_guards_are_not_shape_failures():
    parent = nn.Sequential(nn.Sequential(nn.ReLU()))
    with pytest.raises(BudgetExceeded):
        prepare(parent, input_spec=TensorSpec((2, 4)), limits=Limits(max_depth=2))
    with pytest.raises(BudgetExceeded):
        prepare(
            nn.Sequential(nn.ReLU(), nn.Identity()),
            input_spec=TensorSpec((2, 4)),
            limits=Limits(max_nodes=1),
        )


def test_inference_mode_parameters_are_not_admitted_as_trainable_state():
    with torch.inference_mode():
        parent = nn.Linear(4, 4)
    with pytest.raises(UnsupportedConfiguration):
        prepare(parent, input_spec=TensorSpec((2, 4)))
