"""Small data-only helpers shared by the PyTorch adapter."""

import json
from collections.abc import Mapping
from typing import Any

from .types import UnsafeManifestError

_SHAPE_KEYS = frozenset({"shape", "other_shape", "last_im_shape"})


def data_copy(value: Any, *, context: str) -> Any:
    """Return a JSON-safe deep copy, rejecting executable/opaque metadata."""
    try:
        return json.loads(json.dumps(value, allow_nan=False, separators=(",", ":")))
    except (TypeError, ValueError) as error:
        raise UnsafeManifestError(f"{context} must contain only finite JSON data") from error


def data_mapping(value: Any, *, context: str) -> dict[str, Any]:
    """Validate and copy a data-only mapping."""
    if not isinstance(value, Mapping):
        raise UnsafeManifestError(f"{context} must be a mapping")
    copied = data_copy(dict(value), context=context)
    if not isinstance(copied, dict):
        raise UnsafeManifestError(f"{context} must encode to an object")
    return copied


def same_data(left: Any, right: Any) -> bool:
    """Compare portable values after their strict JSON normalization."""
    return data_copy(left, context="data") == data_copy(right, context="data")


def restore_input_spec(value: Mapping[str, Any]) -> dict[str, Any]:
    """Restore shape fields to the exact values grammar callbacks expect."""
    import torch

    copied = data_mapping(value, context="input_spec")
    for key in _SHAPE_KEYS:
        if isinstance(copied.get(key), list):
            copied[key] = torch.Size(copied[key])
    return copied
