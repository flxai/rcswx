"""Framework-free, lossless portable architecture values.

Portable architecture data deliberately contains only JSON values.  Python objects,
callbacks and framework state are retained by the legacy adapter, never smuggled
through this serialization boundary.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from . import _core

_JSON = dict[str, Any] | list[Any] | str | int | float | bool | None


def _encode(value: object) -> str:
    """Encode JSON data and have Rust validate the full architecture contract."""
    if not isinstance(value, Mapping):
        raise TypeError("architecture must be a mapping")
    try:
        text = json.dumps(value, allow_nan=False, separators=(",", ":"))
    except (TypeError, ValueError) as error:
        raise ValueError("portable architecture data must be JSON") from error
    return _core.validate_architecture(text)


@dataclass(frozen=True, slots=True, init=False)
class Architecture:
    """An immutable, versioned structural tree with no framework dependency."""

    _json: str

    def __init__(self, value: Mapping[str, _JSON] | str):
        text = _core.validate_architecture(value) if isinstance(value, str) else _encode(value)
        object.__setattr__(self, "_json", text)

    @classmethod
    def from_dict(cls, mapping: Mapping[str, _JSON]) -> Architecture:
        return cls(mapping)

    @classmethod
    def from_json(cls, text: str) -> Architecture:
        if not isinstance(text, str):
            raise TypeError("architecture JSON must be a string")
        return cls(text)

    @classmethod
    def _from_validated_json(cls, text: str) -> Architecture:
        """Wrap Rust-validated output, never unchecked caller-supplied JSON."""
        architecture = object.__new__(cls)
        object.__setattr__(architecture, "_json", text)
        return architecture

    @classmethod
    def from_tree(
        cls,
        description: Sequence[object],
        *,
        grammar: str = "einspace",
        grammar_version: str = "1",
        input_spec: _JSON = None,
    ) -> Architecture:
        """Build a portable tree from ``(operation_name, *children)`` descriptions."""
        if not isinstance(grammar, str) or not isinstance(grammar_version, str):
            raise TypeError("grammar and grammar_version must be strings")
        nodes: list[dict[str, _JSON]] = []

        def append(node: object) -> int:
            if isinstance(node, (str, bytes)) or not isinstance(node, Sequence) or not node:
                raise ValueError("tree descriptions must be non-empty (operation_name, *children)")
            name = node[0]
            if not isinstance(name, str):
                raise TypeError("operation names must be strings")
            index = len(nodes)
            nodes.append(
                {
                    "id": str(index),
                    "name": name,
                    "children": [],
                    "parameters": None,
                    "provenance": None,
                }
            )
            children = [append(child) for child in node[1:]]
            nodes[index]["children"] = children
            return index

        root = append(description)
        return cls.from_dict(
            {
                "schema": 1,
                "grammar": grammar,
                "grammar_version": grammar_version,
                "root": root,
                "nodes": nodes,
                "input_spec": input_spec,
            }
        )

    def to_dict(self) -> dict[str, _JSON]:
        return json.loads(self._json)

    def to_json(self) -> str:
        return self._json

    def __repr__(self) -> str:
        value = self.to_dict()
        return (
            "Architecture("
            f"grammar={value['grammar']!r}, grammar_version={value['grammar_version']!r}, "
            f"nodes={len(value['nodes'])})"
        )


def from_dict(mapping: Mapping[str, _JSON]) -> Architecture:
    """Create an :class:`Architecture` from its versioned wire representation."""
    return Architecture.from_dict(mapping)


def from_json(text: str) -> Architecture:
    """Create an :class:`Architecture` from its validated JSON representation."""
    return Architecture.from_json(text)


def to_dict(architecture: Architecture) -> dict[str, _JSON]:
    """Return an independent mapping for a portable architecture."""
    if not isinstance(architecture, Architecture):
        raise TypeError("expected an Architecture")
    return architecture.to_dict()


def to_json(architecture: Architecture) -> str:
    """Return validated JSON, preserving the spelling of opaque metadata values."""
    if not isinstance(architecture, Architecture):
        raise TypeError("expected an Architecture")
    return architecture.to_json()
