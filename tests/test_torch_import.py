"""Consumer-visible contracts for strict, grammar-bound PyTorch import."""

from copy import deepcopy
from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F
from rcswx import Architecture, edit_path
from rcswx.torch import UnsupportedModuleError, build, capture, import_model
from rcswx.torch.grammar_backends import EinspaceBackend
from torch import nn


@pytest.fixture(autouse=True)
def isolated_randomness():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(21)
        yield


class Functional(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(8, 16)

    def forward(self, x):
        return F.relu(F.linear(x, self.layer.weight, self.layer.bias))


class Residual(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Linear(16, 16)
        self.second = nn.Linear(16, 16)

    def forward(self, x):
        first = torch.relu(self.first(x)) + x
        return self.second(first).relu() + first


class Concatenate(nn.Module):
    def __init__(self):
        super().__init__()
        self.left = nn.Linear(8, 16)
        self.right = nn.Linear(8, 32)

    def forward(self, x):
        return torch.cat((self.left(x), self.right(x)), dim=1)


def test_module_and_functional_spellings_have_identical_architectural_meaning():
    inputs = (torch.randn(4, 8),)
    first = import_model(Functional(), example_inputs=inputs)
    second = import_model(
        nn.Sequential(nn.Linear(8, 16), nn.ReLU()),
        example_inputs=inputs,
        grammar="einspace",
        grammar_version="1",
    )
    assert first.architecture == second.architecture
    assert edit_path(first.architecture, second.architecture).distance == 0


def test_parameter_changes_remain_visible_to_existing_edit_costs():
    inputs = (torch.randn(4, 8),)
    first = import_model(nn.Linear(8, 16), example_inputs=inputs)
    second = import_model(nn.Linear(8, 32), example_inputs=inputs)
    plan = edit_path(first.architecture, second.architecture)
    assert plan.distance == 0.25
    assert plan.select("1").cost == 0.25


@pytest.mark.parametrize(
    "family",
    [
        "dense",
        "residual",
        "concatenation",
        "normalization",
        "layer_norm",
        "identity",
        "convolution",
    ],
)
def test_reconstruction_preserves_forward_gradients_and_running_state(family):
    if family == "dense":
        source = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 32), nn.Softmax(dim=-1))
        shape = (4, 8)
    elif family == "residual":
        source, shape = Residual(), (4, 16)
    elif family == "concatenation":
        source, shape = Concatenate(), (4, 8)
    elif family == "normalization":
        source, shape = nn.Sequential(nn.BatchNorm1d(16), nn.ReLU()), (4, 16, 7)
    elif family == "layer_norm":
        source, shape = nn.Sequential(nn.LayerNorm(16), nn.ReLU()), (4, 16)
    elif family == "identity":
        source, shape = nn.Identity(), (4, 8)
    else:
        source = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU())
        shape = (4, 3, 8, 8)
    source = source.to(dtype=torch.float64)
    example = torch.randn(shape, dtype=torch.float64)
    imported = import_model(source, example_inputs=(example,))
    rebuilt = build(imported)
    # State transfer is confined to this differential verifier, never crossover.
    with torch.no_grad():
        for original, target in zip(source.parameters(), rebuilt.parameters(), strict=True):
            target.copy_(original.reshape_as(target))
        for original, target in zip(source.buffers(), rebuilt.buffers(), strict=True):
            target.copy_(original)
    first = example.clone().requires_grad_()
    second = example.clone().requires_grad_()
    expected, actual = source(first), rebuilt(second)
    torch.testing.assert_close(actual, expected, rtol=1e-8, atol=1e-10)
    expected.square().mean().backward()
    actual.square().mean().backward()
    torch.testing.assert_close(second.grad, first.grad, rtol=1e-8, atol=1e-10)
    for original, target in zip(source.parameters(), rebuilt.parameters(), strict=True):
        torch.testing.assert_close(
            target.grad, original.grad.reshape_as(target), rtol=1e-8, atol=1e-10
        )
    for original, target in zip(source.buffers(), rebuilt.buffers(), strict=True):
        torch.testing.assert_close(target, original, rtol=1e-8, atol=1e-10)


def test_import_preserves_source_inputs_gradients_modes_state_and_rng():
    source = nn.Sequential(nn.BatchNorm1d(16), nn.ReLU())
    source[0].weight.grad = torch.ones_like(source[0].weight)
    inputs = torch.randn(4, 16, 7, requires_grad=True)
    saved_input = inputs.detach().clone()
    saved_state = deepcopy(source.state_dict())
    parameter_ids = [id(parameter) for parameter in source.parameters()]
    modules = tuple(source.modules())
    rng = torch.random.get_rng_state()
    imported = import_model(source, example_inputs=(inputs,))
    assert tuple(source.modules()) == modules
    assert [id(parameter) for parameter in source.parameters()] == parameter_ids
    for key, value in saved_state.items():
        torch.testing.assert_close(source.state_dict()[key], value, rtol=0, atol=0)
    torch.testing.assert_close(source[0].weight.grad, torch.ones_like(source[0].weight))
    torch.testing.assert_close(inputs, saved_input, rtol=0, atol=0)
    assert inputs.grad is None
    assert all(module.training for module in source.modules())
    assert torch.equal(torch.random.get_rng_state(), rng)
    # Import is a snapshot, not a carrier installed on or a reference to the source.
    before = imported.architecture.to_json()
    source[1] = nn.Sigmoid()
    assert imported.architecture.to_json() == before
    with pytest.raises(UnsupportedModuleError):
        capture(source)


@pytest.mark.parametrize(
    "model,shape",
    [
        (lambda: nn.Linear(8, 24), (4, 8)),
        (lambda: nn.Linear(8, 16, bias=False), (4, 8)),
        (lambda: nn.ReLU(inplace=True), (4, 8)),
        (lambda: nn.Softmax(dim=0), (4, 8)),
        (lambda: nn.LayerNorm(16, eps=1e-3), (4, 16)),
        (lambda: nn.LayerNorm(7), (4, 16, 7)),
        (lambda: nn.Conv2d(4, 16, 3, padding=1, groups=2), (4, 4, 8, 8)),
        (lambda: nn.Conv2d(3, 16, 5), (4, 3, 8, 8)),
    ],
)
def test_unrepresentable_options_are_not_approximated(model, shape):
    source = model()
    state = deepcopy(source.state_dict())
    with pytest.raises(UnsupportedModuleError):
        import_model(source, example_inputs=(torch.randn(shape),))
    for key, value in state.items():
        torch.testing.assert_close(source.state_dict()[key], value)


def test_unknown_grammar_and_version_are_rejected_without_fallback():
    source = nn.Linear(8, 16)
    inputs = (torch.randn(4, 8),)
    for options in ({"grammar": "hnasbench201"}, {"grammar_version": "2"}):
        with pytest.raises(UnsupportedModuleError):
            import_model(source, example_inputs=inputs, **options)


def test_crossing_dependencies_and_repeated_state_are_not_duplicated():
    class Crossing(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = nn.Linear(16, 16)

        def forward(self, x):
            a = self.layer(x).relu()
            b = (x + a).relu()
            return a + b

    class Repeated(Crossing):
        def forward(self, x):
            return self.layer(self.layer(x))

    for model in (Crossing(), Repeated()):
        with pytest.raises(UnsupportedModuleError):
            import_model(model, example_inputs=(torch.randn(4, 16),))


def test_input_dependent_control_flow_and_mutation_are_rejected_without_touching_source():
    class Dynamic(nn.Module):
        def forward(self, x):
            return x.relu() if x.sum() > 0 else x.softmax(-1)

    class Mutating(nn.Module):
        def __init__(self):
            super().__init__()
            self.counter = 0

        def forward(self, x):
            self.counter += 1
            return x.relu()

    source = Mutating()
    for model in (Dynamic(), source):
        with pytest.raises(UnsupportedModuleError):
            import_model(model, example_inputs=(torch.randn(4, 16),))
    assert source.counter == 0


def test_training_dependent_graph_changes_fail_faithfulness():
    class ModeDependent(nn.Module):
        def forward(self, x):
            return x.relu() if self.training else x.softmax(-1)

    source = ModeDependent()
    with pytest.raises(UnsupportedModuleError):
        import_model(source, example_inputs=(torch.arange(32.0).reshape(4, 8),))
    assert source.training


def test_numerically_wrong_lowering_is_rejected_even_when_shapes_match(monkeypatch):
    lower = EinspaceBackend.lower

    def wrong_activation(self, graph, spec):
        lowered = lower(self, graph, spec)
        data = lowered.architecture.to_dict()
        for node in data["nodes"]:
            if node["name"] == "relu":
                node["name"] = "softmax"
        return replace(lowered, architecture=Architecture.from_dict(data))

    monkeypatch.setattr(EinspaceBackend, "lower", wrong_activation)
    with pytest.raises(UnsupportedModuleError):
        import_model(nn.ReLU(), example_inputs=(torch.arange(32.0).reshape(4, 8),))


@pytest.mark.parametrize("hook_kind", ["module_pre", "parameter", "post_accumulate"])
def test_unrepresentable_gradient_hooks_are_rejected(hook_kind):
    source = nn.Linear(8, 16)
    if hook_kind == "module_pre":
        handle = source.register_full_backward_pre_hook(
            lambda module, gradients: (torch.zeros_like(gradients[0]),)
        )
    elif hook_kind == "parameter":
        handle = source.weight.register_hook(lambda gradient: gradient * 0)
    else:
        handle = source.weight.register_post_accumulate_grad_hook(
            lambda parameter: parameter.grad.zero_()
        )
    with handle, pytest.raises(UnsupportedModuleError):
        import_model(source, example_inputs=(torch.randn(4, 8),))


def test_forward_mode_mutation_is_rejected_without_touching_source():
    class MutatingMode(nn.Module):
        def forward(self, x):
            self.eval()
            return x.relu()

    source = MutatingMode()
    with pytest.raises(UnsupportedModuleError):
        import_model(source, example_inputs=(torch.randn(4, 8),))
    assert source.training
