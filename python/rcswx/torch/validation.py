"""Strict, isolated validation for the explicit imported-model workflow."""

from collections.abc import Mapping

import torch

from ._isolation import isolated_inputs, isolated_model, preserve_rng
from .build import _reconstruct_tree, build
from .provenance import capture
from .types import UnsupportedModuleError

# Numerical equivalence, not bit identity: convolution routing can change summation order.
_TOLERANCES = {
    torch.float64: (1e-7, 1e-9),
    torch.float32: (1e-4, 1e-5),
    torch.float16: (1e-2, 1e-3),
    torch.bfloat16: (3e-2, 3e-2),
}


def validate_architecture(architecture, input_spec: Mapping) -> dict:
    """Validate the actual grammar derivation without constructing learned tensors."""
    try:
        root = _reconstruct_tree(architecture.to_dict(), input_spec)
        for node in root.serialise():
            if not node.operation.valid(node):
                raise UnsupportedModuleError(
                    f"validity: grammar operation {node.operation.name!r} rejects its input interface"
                )
        shape = list(root.output_params["shape"])
        if not shape or any(not isinstance(dim, int) or dim <= 0 for dim in shape):
            raise UnsupportedModuleError("validity: grammar produces an invalid output shape")
        return {"shape": shape}
    except UnsupportedModuleError:
        raise
    except Exception as error:
        raise UnsupportedModuleError(f"validity: grammar reconstruction failed: {error}") from error


def _output(value, expected, input_tensor, stage):
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise UnsupportedModuleError(f"{stage}: expected one floating tensor output")
    if list(value.shape) != expected["shape"]:
        raise UnsupportedModuleError(
            f"{stage}: output shape {list(value.shape)} differs from inferred {expected['shape']}"
        )
    if value.dtype != input_tensor.dtype or value.device != input_tensor.device:
        raise UnsupportedModuleError(
            f"{stage}: output dtype/device differs from the input contract"
        )
    if not torch.isfinite(value).all():
        raise UnsupportedModuleError(f"{stage}: output contains non-finite values")
    return {"shape": list(value.shape), "dtype": str(value.dtype), "device": str(value.device)}


def _close(actual, expected, stage):
    rtol, atol = _TOLERANCES.get(expected.dtype, (0, 0))
    try:
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    except AssertionError as error:
        raise UnsupportedModuleError(
            f"faithfulness: {stage} differs after reconstruction: {error}"
        ) from error


def _target_roles(paths):
    result = {}
    for path in paths:
        role = path.rsplit(".", 1)[-1]
        if role in result:
            raise UnsupportedModuleError(
                f"faithfulness: ambiguous reconstructed state role {role!r}"
            )
        result[role] = path
    return result


def _transfer_state(source, rebuilt, lowering):
    """Return verified state pairs; source paths never become child checkpoint keys."""
    occurrences = capture(rebuilt).bindings["occurrences"]
    source_parameters = dict(source.named_parameters())
    source_buffers = dict(source.named_buffers())
    target_parameters = dict(rebuilt.named_parameters())
    target_buffers = dict(rebuilt.named_buffers())
    seen_source, seen_target = set(), set()
    buffer_pairs = []
    with torch.no_grad():
        for binding in lowering.state_bindings:
            occurrence = occurrences[str(binding.occurrence)]
            for kind, bindings, sources, targets in (
                ("parameter", binding.parameters, source_parameters, target_parameters),
                ("buffer", binding.buffers, source_buffers, target_buffers),
            ):
                roles = _target_roles(occurrence[f"{kind}_paths"])
                if roles.keys() != bindings.keys():
                    raise UnsupportedModuleError(
                        "faithfulness: reconstruction changes parameterization/state"
                    )
                for role, source_path in bindings.items():
                    target_path = roles[role]
                    if (kind, source_path) in seen_source or (kind, target_path) in seen_target:
                        raise UnsupportedModuleError(
                            "faithfulness: state correspondence is not one-to-one"
                        )
                    if source_path not in sources or target_path not in targets:
                        raise UnsupportedModuleError(
                            "faithfulness: state correspondence refers to absent state"
                        )
                    original, target = sources[source_path], targets[target_path]
                    value = original
                    if binding.reshape_weight and kind == "parameter" and role == "weight":
                        if (
                            original.ndim != 4
                            or target.ndim != 2
                            or original.numel() != target.numel()
                        ):
                            raise UnsupportedModuleError(
                                "faithfulness: invalid convolution weight correspondence"
                            )
                        value = original.reshape_as(target)
                    if value.shape != target.shape or value.dtype != target.dtype:
                        raise UnsupportedModuleError("faithfulness: state shape or dtype differs")
                    target.copy_(value)
                    if kind == "parameter":
                        target.requires_grad_(original.requires_grad)
                    else:
                        buffer_pairs.append((source_path, target_path))
                    seen_source.add((kind, source_path))
                    seen_target.add((kind, target_path))
    expected_source = {("parameter", path) for path in source_parameters} | {
        ("buffer", path) for path in source_buffers
    }
    expected_target = {("parameter", path) for path in target_parameters} | {
        ("buffer", path) for path in target_buffers
    }
    if seen_source != expected_source or seen_target != expected_target:
        raise UnsupportedModuleError(
            "faithfulness: unrepresented source or reconstructed parameters/buffers"
        )
    return buffer_pairs


def verify_import(source, lowering, inputs: tuple[torch.Tensor, ...]) -> dict:
    """Check source/reconstruction behavior under corresponding state, without inheritance."""
    tensors = (*source.parameters(), *source.buffers(), *inputs)
    try:
        with preserve_rng(tensors):
            expected = validate_architecture(
                lowering.architecture, lowering.architecture.to_dict()["input_spec"]
            )
            reference = isolated_model(source)
            rebuilt = build(
                lowering.architecture,
                build_options={
                    "dtype": inputs[0].dtype,
                    "device": inputs[0].device,
                    "training": source.training,
                },
            )
            buffer_pairs = _transfer_state(reference, rebuilt, lowering)
            modes = (source.training, not source.training)
            for training in modes:
                reference.train(training)
                rebuilt.train(training)
                with torch.no_grad():
                    original_output = reference(*isolated_inputs(inputs))
                    output = rebuilt(*isolated_inputs(inputs))
                signature = _output(original_output, expected, inputs[0], "faithfulness source")
                _output(output, expected, inputs[0], "faithfulness reconstruction")
                _close(output, original_output, f"output with training={training}")
                for source_path, target_path in buffer_pairs:
                    _close(
                        rebuilt.get_buffer(target_path),
                        reference.get_buffer(source_path),
                        f"buffer {source_path!r}",
                    )
            rtol, atol = _TOLERANCES[inputs[0].dtype]
            return {
                "checks": ["grammar", "state_correspondence", "forward", "buffer_updates"],
                "output": signature,
                "rtol": rtol,
                "atol": atol,
                "training_modes": list(modes),
            }
    except UnsupportedModuleError:
        raise
    except Exception as error:
        raise UnsupportedModuleError(f"faithfulness: verification failed: {error}") from error


def validate_runtime(model, inputs: tuple[torch.Tensor, ...], expected: Mapping) -> dict:
    """Exercise a disposable child copy; never alter the returned model's state."""
    try:
        with preserve_rng((*model.parameters(), *model.buffers(), *inputs)):
            disposable = isolated_model(model)
            with torch.no_grad():
                output = disposable(*isolated_inputs(inputs))
            signature = _output(output, expected, inputs[0], "validity runtime")
            return {"checks": ["grammar", "runtime"], "output": signature}
    except UnsupportedModuleError:
        raise
    except Exception as error:
        raise UnsupportedModuleError(
            f"validity runtime: child execution failed: {error}"
        ) from error
