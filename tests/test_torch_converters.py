"""Tests for exact-type converter dispatch and explicit import boundaries."""

import pytest
import torch
from rcswx.torch import (
    UnsupportedModuleError,
    build,
    capture,
    linear_relu_sequential_converter,
    register_importer,
    unregister_importer,
)
from torch import nn

INPUT_SPEC = {
    "shape": [3, 8],
    "mode": "col",
    "other_shape": None,
    "other_mode": None,
    "branching_factor": 1,
    "last_im_shape": None,
}


def test_registered_linear_relu_converter_captures_without_running_forward():
    class ForwardForbidden(nn.Sequential):
        def forward(self, value):
            raise AssertionError("capture must not execute forward")

    registration = register_importer(
        "forward-forbidden-linear-relu",
        "1",
        [ForwardForbidden],
        linear_relu_sequential_converter,
    )
    try:
        source = ForwardForbidden(nn.Linear(8, 16), nn.ReLU())
        captured = capture(source, input_spec=INPUT_SPEC)
        rebuilt = build(captured)
    finally:
        unregister_importer(registration.name, registration.version)

    assert captured.metadata["importer"] == {
        "name": "forward-forbidden-linear-relu",
        "version": "1",
    }
    with torch.no_grad():
        assert tuple(rebuilt(torch.zeros(3, 8)).shape) == (3, 16)


def test_converter_dispatch_is_exact_type_and_rejects_unsupported_subclasses():
    class SequentialSubclass(nn.Sequential):
        pass

    registration = register_importer(
        "base-linear-relu",
        "1",
        [nn.Sequential],
        linear_relu_sequential_converter,
    )
    try:
        with pytest.raises(UnsupportedModuleError, match="exact-type"):
            capture(SequentialSubclass(nn.Linear(8, 16), nn.ReLU()), input_spec=INPUT_SPEC)
    finally:
        unregister_importer(registration.name, registration.version)


def test_converter_dispatch_rejects_ambiguous_highest_priority_registration():
    first = register_importer(
        "linear-relu-a", "1", [nn.Sequential], linear_relu_sequential_converter
    )
    second = register_importer(
        "linear-relu-b", "1", [nn.Sequential], linear_relu_sequential_converter
    )
    try:
        with pytest.raises(UnsupportedModuleError, match="ambiguous"):
            capture(nn.Sequential(nn.Linear(8, 16), nn.ReLU()), input_spec=INPUT_SPEC)
    finally:
        unregister_importer(second.name, second.version)
        unregister_importer(first.name, first.version)


def test_converter_rejects_parameter_sharing_instead_of_forcing_a_tree():
    shared = nn.Linear(8, 8)
    source = nn.Sequential(shared, shared)

    with pytest.raises(UnsupportedModuleError, match="sharing"):
        capture(source, input_spec=INPUT_SPEC)
