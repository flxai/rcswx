"""Native typed, ordered semantic-tree diagnostics without a framework dependency."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, Literal, TypedDict

from . import _core


class Correspondence(TypedDict):
    first_id: str
    second_id: str


class TypedEdit(TypedDict):
    operation: Literal["match", "substitute", "insert", "delete"]
    first_id: str | None
    second_id: str | None
    cost: int


class TypedComparison(TypedDict):
    version: str
    tree_version: str
    policy: str
    backend: str
    direction: Literal["second-to-first"]
    distance: int
    correspondences: list[Correspondence]
    edits: list[TypedEdit]


def _encode(tree: Mapping[str, Any]) -> str:
    if not isinstance(tree, Mapping):
        raise TypeError("typed trees must be mappings")
    try:
        return json.dumps(tree, allow_nan=False, separators=(",", ":"))
    except (TypeError, ValueError) as error:
        raise ValueError("typed tree data must be JSON") from error


def compare_typed(first: Mapping[str, Any], second: Mapping[str, Any]) -> TypedComparison:
    """Return one native minimum-cost path transforming ``second`` toward ``first``.

    Both mappings must satisfy ``rcswx-typed-ordered-v1``. Rust validates the
    occurrence trees and runs the ordered recursive ``typed-unit-v1`` policy;
    Python only transports JSON. IDs never score. Every semantic occurrence is
    represented by one match/substitute/insert/delete, with a cost of zero for
    matches and one otherwise. Sum of edit costs equals the native distance.
    Correspondences preserve immediate parents and order. Insertions/deletions
    consume whole subtrees; no retained child is promoted or newly wrapped.

    This is a structural diagnostic, not semantic approval or a relevance metric.
    Unlike the einspace API, it does not permute branches or perform crossover.
    """
    return json.loads(_core.compare_typed_json(_encode(first), _encode(second)))
