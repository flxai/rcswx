"""Allowlisted, exact-type converter registration for module capture."""

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from torch import nn

from .types import ConverterResult, UnsupportedModuleError

Converter = Callable[[Any, Any | None], ConverterResult]


@dataclass(frozen=True, slots=True)
class ImporterRegistration:
    """One installed converter declaration; no import paths are persisted."""

    name: str
    version: str
    supported_types: tuple[type, ...]
    converter: Converter
    priority: int


_REGISTRY: list[ImporterRegistration] = []


def register_importer(
    name: str,
    version: str,
    supported_types: Iterable[type],
    converter: Converter,
    *,
    priority: int = 0,
) -> ImporterRegistration:
    """Register a converter for exact module types.

    Matching uses ``type(model) is registered_type`` rather than ``isinstance``
    so a subclass cannot silently inherit a converter for different forward
    semantics. A tie at the highest priority is rejected at dispatch time.
    """
    if not isinstance(name, str) or not name:
        raise ValueError("importer name must be a non-empty string")
    if not isinstance(version, str) or not version:
        raise ValueError("importer version must be a non-empty string")
    if not callable(converter):
        raise TypeError("converter must be callable")
    if not isinstance(priority, int):
        raise TypeError("importer priority must be an integer")

    types = tuple(supported_types)
    if not types or any(not isinstance(item, type) for item in types):
        raise TypeError("supported_types must contain at least one concrete type")
    if any(not issubclass(item, nn.Module) for item in types):
        raise TypeError("supported_types must contain torch.nn.Module types")
    if any(item.name == name and item.version == version for item in _REGISTRY):
        raise ValueError(f"importer {name!r} version {version!r} is already registered")

    registration = ImporterRegistration(name, version, types, converter, priority)
    _REGISTRY.append(registration)
    return registration


def unregister_importer(name: str, version: str) -> None:
    """Remove one exact importer registration, primarily for isolated tests."""
    for index, registration in enumerate(_REGISTRY):
        if registration.name == name and registration.version == version:
            del _REGISTRY[index]
            return
    raise KeyError(f"no importer {name!r} version {version!r} is registered")


def registered_importers() -> tuple[ImporterRegistration, ...]:
    """Return registrations in deterministic declaration order."""
    return tuple(_REGISTRY)


def resolve_importer(model: Any, importer: str | None = None) -> ImporterRegistration:
    """Select the single highest-priority converter for ``type(model)``."""
    exact_type = type(model)
    candidates = [
        registration
        for registration in _REGISTRY
        if exact_type in registration.supported_types
        and (importer is None or registration.name == importer)
    ]
    if not candidates:
        requested = f" importer {importer!r}" if importer is not None else ""
        raise UnsupportedModuleError(
            f"{exact_type.__module__}.{exact_type.__qualname__} has no retained RCSWX "
            f"provenance or registered exact-type{requested} converter"
        )

    priority = max(registration.priority for registration in candidates)
    selected = [registration for registration in candidates if registration.priority == priority]
    if len(selected) != 1:
        names = ", ".join(
            f"{registration.name}@{registration.version}"
            for registration in sorted(selected, key=lambda item: (item.name, item.version))
        )
        raise UnsupportedModuleError(
            f"ambiguous exact-type converters for "
            f"{exact_type.__module__}.{exact_type.__qualname__}: {names}"
        )
    return selected[0]
