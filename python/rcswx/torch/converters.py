"""Narrow, opt-in converter examples for the module capture registry."""

from collections.abc import Mapping
from typing import Any

import torch.nn as nn

from .types import ConverterResult, UnsupportedModuleError


def linear_relu_sequential_converter(
    model: nn.Sequential, input_spec: Mapping[str, Any] | None
) -> ConverterResult:
    """Import exactly ``nn.Sequential(nn.Linear, nn.ReLU)`` into einspace.

    This intentionally does not claim to import arbitrary ``nn.Sequential``
    modules. The caller must register it for ``nn.Sequential`` explicitly;
    subclasses are rejected by exact-type dispatch.
    """
    if len(model) != 2 or type(model[0]) is not nn.Linear or type(model[1]) is not nn.ReLU:
        raise UnsupportedModuleError(
            "linear_relu_sequential_converter supports exactly nn.Sequential(nn.Linear, nn.ReLU)"
        )
    if model[0].bias is None or model[1].inplace:
        raise UnsupportedModuleError(
            "the einspace linear/ReLU import requires a biased, non-inplace source module"
        )
    if input_spec is None:
        raise UnsupportedModuleError(
            "the linear/ReLU converter requires an explicit grammar input_spec"
        )
    try:
        input_width = input_spec["shape"][-1]
    except (KeyError, TypeError, IndexError) as error:
        raise UnsupportedModuleError(
            "converter input_spec must include a non-empty shape"
        ) from error
    if input_width != model[0].in_features:
        raise UnsupportedModuleError(
            "converter input_spec width does not match nn.Linear.in_features"
        )

    from rcswx.portable import Architecture

    architecture = Architecture.from_tree(
        (
            "sequential",
            ("computation", (f"linear({model[0].out_features})",)),
            ("computation", ("relu",)),
        ),
        grammar="einspace",
        grammar_version="1",
        input_spec=dict(input_spec),
    )
    bindings = {
        "0": {"module_paths": [""], "virtual": True},
        "1": {"module_paths": [], "virtual": True},
        "2": {"module_paths": ["0"]},
        "3": {"module_paths": [], "virtual": True},
        "4": {"module_paths": ["1"]},
    }
    return ConverterResult(
        architecture=architecture,
        input_spec=dict(input_spec),
        conventions={"source_family": "torch.nn.Sequential(nn.Linear, nn.ReLU)"},
        bindings=bindings,
    )
