"""Observable contracts for capture/build and module-level crossover."""

from copy import deepcopy

import pytest
import torch
from rcswx.portable import Architecture
from rcswx.torch import (
    BuildOptions,
    StaleProvenanceError,
    UnsafeManifestError,
    UnsupportedModuleError,
    build,
    capture,
    crossover_with_report,
    load,
    save,
)

INPUT_SPEC = {
    "shape": [4, 8],
    "mode": "col",
    "other_shape": None,
    "other_mode": None,
    "branching_factor": 1,
    "last_im_shape": None,
}
BUFFER_INPUT_SPEC = {
    "shape": [4, 8, 3],
    "mode": "col",
    "other_shape": None,
    "other_mode": None,
    "branching_factor": 1,
    "last_im_shape": None,
}


def architecture(activation: str) -> Architecture:
    return Architecture.from_tree(
        (
            "sequential",
            ("computation", ("linear(16)",)),
            ("computation", (activation,)),
        ),
        grammar="einspace",
        grammar_version="1",
        input_spec=INPUT_SPEC,
    )


def normalized_architecture() -> Architecture:
    return Architecture.from_tree(
        (
            "sequential",
            ("computation", ("norm",)),
            ("computation", ("norm",)),
        ),
        grammar="einspace",
        grammar_version="1",
        input_spec=BUFFER_INPUT_SPEC,
    )


def tensor_state(model):
    return {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
        if isinstance(value, torch.Tensor)
    }


def assert_same_tensor_state(actual, expected):
    assert actual.keys() == expected.keys()
    for name, value in actual.items():
        assert torch.equal(value, expected[name]), name


def test_module_crossover_captures_portably_and_leaves_parents_unchanged():
    first = build(architecture("relu"))
    second = build(architecture("softmax"))
    first.train(False)
    second.train(True)
    tracked_parameter = next(first.parameters())
    tracked_parameter.grad = torch.full_like(tracked_parameter, 3)
    first_before = tensor_state(first)
    second_before = tensor_state(second)
    parent_parameter_ids = {id(parameter) for parameter in first.parameters()} | {
        id(parameter) for parameter in second.parameters()
    }

    result = crossover_with_report(first, second, seed=19)

    assert result.child is not first
    assert result.child is not second
    assert_same_tensor_state(tensor_state(first), first_before)
    assert_same_tensor_state(tensor_state(second), second_before)
    assert torch.equal(tracked_parameter.grad, torch.full_like(tracked_parameter, 3))
    assert all(id(parameter) not in parent_parameter_ids for parameter in result.child.parameters())
    assert not first.training
    assert not result.child.training
    assert second.training
    recaptured = capture(result.child)
    assert recaptured.architecture == result.architecture
    assert result.report["module_architecture"] == result.architecture.to_dict()
    assert result.report["module_capture"]["child"]["format"] == "rcswx.torch.manifest"
    with torch.no_grad():
        assert tuple(result.child(torch.zeros(4, 8)).shape) == (4, 16)


def test_build_and_crossover_preserve_captured_execution_until_explicitly_overridden():
    options = BuildOptions(device="cpu", dtype=torch.float64, training=False)
    first = build(architecture("relu"), build_options=options)
    second = build(architecture("softmax"), build_options=options)

    rebuilt = build(capture(first))
    result = crossover_with_report(first, second, seed=7)
    overridden = build(capture(first), build_options={"dtype": torch.float32, "training": True})

    assert not rebuilt.training
    assert not result.child.training
    assert rebuilt is not first
    assert {parameter.dtype for parameter in rebuilt.parameters()} == {torch.float64}
    assert {parameter.dtype for parameter in result.child.parameters()} == {torch.float64}
    assert overridden.training
    assert {parameter.dtype for parameter in overridden.parameters()} == {torch.float32}


def test_build_honours_explicit_dtype_device_and_training_options():
    model = build(
        architecture("relu"),
        build_options=BuildOptions(device="cpu", dtype=torch.float64, training=False),
    )

    assert not model.training
    assert {parameter.device.type for parameter in model.parameters()} == {"cpu"}
    assert {parameter.dtype for parameter in model.parameters()} == {torch.float64}


def test_build_rejects_unregistered_portable_parameters_instead_of_dropping_them():
    data = architecture("relu").to_dict()
    data["nodes"][0]["parameters"] = {"unknown": "value"}

    with pytest.raises(UnsupportedModuleError, match="parameters unsupported"):
        build(Architecture.from_dict(data))


def test_capture_rejects_shared_buffers_instead_of_losing_their_identity():
    model = build(normalized_architecture())
    first_norm = model.first_fn.computation_fn.fn
    second_norm = model.second_fn.computation_fn.fn
    second_norm.running_mean = first_norm.running_mean

    with pytest.raises(UnsupportedModuleError, match="buffer sharing"):
        capture(model)


def test_capture_rejects_distinct_views_with_shared_buffer_storage():
    model = build(normalized_architecture())
    first_norm = model.first_fn.computation_fn.fn
    second_norm = model.second_fn.computation_fn.fn
    second_norm.running_mean = first_norm.running_mean.detach()

    with pytest.raises(UnsupportedModuleError, match="storage sharing"):
        capture(model)


def test_capture_build_and_checkpoint_retain_local_modes_and_tensor_dtypes(tmp_path):
    model = build(normalized_architecture(), build_options={"dtype": torch.float64})
    model.first_fn.computation_fn.fn.eval()
    # Integer state is independent of Module.to's floating-point dtype policy.
    model.second_fn.computation_fn.fn.num_batches_tracked = (
        model.second_fn.computation_fn.fn.num_batches_tracked.to(torch.int32)
    )
    captured = capture(model)
    rebuilt = build(captured)
    assert rebuilt.first_fn.computation_fn.fn.training is False
    assert rebuilt.second_fn.computation_fn.fn.training is True
    assert rebuilt.second_fn.computation_fn.fn.num_batches_tracked.dtype == torch.int32
    rebuilt(torch.randn(*BUFFER_INPUT_SPEC["shape"], dtype=torch.float64)).sum().backward()
    assert rebuilt.first_fn.computation_fn.fn.num_batches_tracked.item() == 0
    assert rebuilt.second_fn.computation_fn.fn.num_batches_tracked.item() == 1

    checkpoint = tmp_path / "local-execution.pt"
    save(model, checkpoint)
    restored = load(checkpoint)
    assert restored.first_fn.computation_fn.fn.training is False
    assert restored.second_fn.computation_fn.fn.num_batches_tracked.dtype == torch.int32
    overridden = build(captured, build_options={"training": False, "dtype": torch.float32})
    assert overridden.second_fn.computation_fn.fn.training is False
    assert overridden.first_fn.computation_fn.fn.weight.dtype == torch.float32


def test_managed_checkpoint_rejects_dtype_conversion_before_loading_state(tmp_path):
    model = build(architecture("relu"), build_options={"dtype": torch.float64})
    checkpoint = tmp_path / "fp64.pt"
    save(model, checkpoint)

    with pytest.raises(UnsafeManifestError, match="dtype mismatch"):
        load(checkpoint, build_options={"dtype": torch.float32})


def test_managed_checkpoint_rejects_forged_embedded_manifest_before_loading(tmp_path):
    model = build(architecture("relu"))
    checkpoint = tmp_path / "valid.pt"
    forged = tmp_path / "forged.pt"
    save(model, checkpoint)
    payload = torch.load(checkpoint, weights_only=True)
    payload["state_dict"]["_rcswx_provenance._extra_state"]["architecture"]["nodes"][4]["name"] = (
        "softmax"
    )
    torch.save(payload, forged)

    with pytest.raises(UnsafeManifestError, match="manifests differ"):
        load(forged)


def test_training_deepcopy_and_managed_checkpoint_preserve_recapturable_provenance(tmp_path):
    model = build(architecture("relu"))
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(1)
    model.train(True)

    trained_capture = capture(model)
    copied = deepcopy(model)
    copied_capture = capture(copied)
    assert copied_capture.architecture == trained_capture.architecture
    assert copied_capture.metadata["execution"]["training"] is True

    checkpoint = tmp_path / "module-state.pt"
    save(model, checkpoint)
    restored = load(checkpoint, build_options={"device": "cpu"})
    assert capture(restored).architecture == trained_capture.architecture
    assert_same_tensor_state(tensor_state(restored), tensor_state(model))


def test_capture_rejects_stale_structure_without_treating_trained_values_as_stale():
    model = build(architecture("relu"))
    with torch.no_grad():
        next(model.parameters()).mul_(0)
    capture(model)

    model.first_fn = torch.nn.Identity()
    with pytest.raises(StaleProvenanceError, match="stale"):
        capture(model)
