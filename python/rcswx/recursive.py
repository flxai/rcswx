# Derived from einsearch 7e713c7951397a6b12bd57638409774a74381746.
# Copyright (c) 2024 Adri Gómez Martín, Felix Möller, Linus Ericsson, Aaron Klein.
# Distributed under the MIT license; see LICENSE.einsearch.
"""Compatibility views over the native retained RCSWX edit plan.

All structural preparation, alignment, dependency construction and application live
in the Rust core.  This module deliberately only snapshots legacy derivation trees,
exposes historical operation objects, and replays the core's declarative object
materialization recipe.
"""

from __future__ import annotations

import copy
import json
import secrets
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from . import _core
from .portable import Architecture


class MatrixOperation:
    """Historical mutable-looking operation view backed by an immutable plan edit."""

    def __init__(
        self,
        op_id=None,
        op_type=None,
        node1_id=None,
        node2_id=None,
        i=None,
        j=None,
        ii=None,
        jj=None,
        value=0,
        disabler_ops=None,
        enabler_ops=None,
    ):
        self.id = op_id
        self.op_type = op_type
        self.i = i
        self.j = j
        self.ii = ii
        self.jj = jj
        self.node1_id = node1_id
        self.node2_id = node2_id
        self.value = value
        self.i_swapped = False
        self.j_swapped = False
        self.disabler_ops = [] if disabler_ops is None else disabler_ops
        self.enabler_ops = [] if enabler_ops is None else enabler_ops

    def __str__(self):
        string = f"{self.op_type} (id: {self.id}) with a cost of {self.value}. "
        if self.disabler_ops:
            string = string[:-2] + " (disabled by "
            for branch in self.disabler_ops:
                for op in branch if isinstance(branch, list) else [branch]:
                    string += f"{op.id}, "
            string = string[:-2] + ")."
        if self.enabler_ops:
            string = string[:-2] + (" (" if not self.disabler_ops else "; ") + "enabled by "
            for branch in self.enabler_ops:
                for op in branch if isinstance(branch, list) else [branch]:
                    string += f"{op.id}, "
            string = string[:-2] + ")."
        return string

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return self.id == other.id


@dataclass(frozen=True, slots=True)
class Selection:
    """An explicitly plan-bound subset of selected-path operation occurrences."""

    _owner: Alignment
    _indices: tuple[int, ...]
    _cost: float | None = None

    @property
    def plan_id(self) -> str:
        return self._owner.id

    @property
    def indices(self) -> tuple[int, ...]:
        return self._indices

    @property
    def operations(self) -> tuple[MatrixOperation, ...]:
        return tuple(self._owner._operation_at(index) for index in self._indices)

    @property
    def cost(self) -> float:
        if self._cost is not None and self._indices:
            return self._cost
        # The legacy raw result preserves Python's integer zero for an empty sum.
        return sum(operation.value for operation in self.operations)


class _PreparedOperation:
    def __init__(self, name: str):
        self.name = name


class _PreparedNode:
    def __init__(self, identifier, name: str, children: list[str], parent_arity: int):
        self.id = identifier
        self.operation = _PreparedOperation(name)
        self.children = children
        self.parent_arity = parent_arity
        self.parent = None

    def is_root(self):
        return self.parent is not None

    def __eq__(self, other):
        return isinstance(other, _PreparedNode) and self.id == other.id

    def __str__(self):
        return f"{self.operation.name} (id {self.id})"

    __repr__ = __str__


@dataclass(slots=True)
class _LegacySnapshot:
    root: object
    nodes: tuple[object, ...]
    architecture: Architecture
    origins: tuple[int, ...]

    @classmethod
    def from_root(cls, root: object, origins: dict[int, int] | None = None) -> _LegacySnapshot:
        occurrences: list[object] = []
        records: list[dict[str, Any]] = []
        if origins is None:
            origins = {}

        def append(node: object, ancestors: tuple[int, ...]) -> int:
            marker = id(node)
            if marker in ancestors:
                raise ValueError("legacy trees must not contain parent/child cycles")
            try:
                name = node.operation.name
                children = node.children
                identifier = str(node.id)
            except AttributeError as error:
                raise TypeError("expected a legacy derivation tree") from error
            index = len(occurrences)
            occurrences.append(node)
            records.append({})
            origins.setdefault(marker, len(origins))
            child_indices = [append(child, (*ancestors, marker)) for child in children]
            records[index] = {
                "id": identifier,
                "name": name,
                "children": child_indices,
                "parameters": None,
                "provenance": None,
            }
            return index

        root_index = append(root, ())
        return cls(
            root,
            tuple(occurrences),
            Architecture.from_dict(
                {
                    "schema": 1,
                    "grammar": "einspace",
                    "grammar_version": "1",
                    "root": root_index,
                    "nodes": records,
                    "input_spec": None,
                }
            ),
            tuple(origins[id(node)] for node in occurrences),
        )

    def apply_parent2_ids(self, ids: Sequence[str]) -> None:
        if len(ids) != len(self.nodes):
            raise RuntimeError("native preparation returned the wrong parent2 ID count")
        for node, identifier in zip(self.nodes, ids, strict=True):
            # Reference preparation assigns Python integers, including arbitrary-width IDs.
            node.id = int(identifier)

    def fingerprint(self) -> tuple[tuple[int, str, str, tuple[int, ...]], ...]:
        return tuple(
            (
                id(node),
                str(node.id),
                node.operation.name,
                tuple(id(child) for child in node.children),
            )
            for node in self.nodes
        )


def _plan_id() -> str:
    # Plan identity is not coupled to the structural selection RNG stream.
    return secrets.token_hex(16)


def _same_raw_parent(first: object, second: object) -> bool:
    if isinstance(first, Architecture) or isinstance(second, Architecture):
        if not isinstance(first, Architecture) or not isinstance(second, Architecture):
            return False
        return first == second
    # Preserve the historic short-circuit's equality dispatch and operation-name check.
    return first.serialise() == second.serialise() and all(
        node1.operation.name == node2.operation.name
        for node1, node2 in zip(first.serialise(), second.serialise(), strict=True)
    )


def _check_compatible_architectures(first: Architecture, second: Architecture) -> None:
    left, right = first.to_dict(), second.to_dict()
    if (left["grammar"], left["grammar_version"]) != (
        right["grammar"],
        right["grammar_version"],
    ):
        raise ValueError("portable parents must use the same grammar and grammar version")


def _memory_check(limiter: object | None):
    if limiter is None:
        return None
    return limiter.check_memory_crossover


def _recipe_payload(event: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    if "kind" in event:
        kind = event["kind"]
        return kind, {key: value for key, value in event.items() if key != "kind"}
    if len(event) != 1:
        raise RuntimeError("invalid native materialization recipe event")
    return next(iter(event.items()))


def _make_sequence(template: object, parent: object | None, identifier: str) -> object:
    """Construct the callback-bearing legacy sequence node requested by the recipe."""
    # The legacy implementation has always used the einspace sequential constructor;
    # importing it here keeps the framework-free structural path import-free.
    from .genotype import DerivationTreeNode, Operation
    from .grammars import einspace

    return DerivationTreeNode(
        int(identifier),
        level=template.level,
        parent=parent,
        input_params=template.input_params,
        depth=template.depth,
        limiter=template.limiter,
        operation=Operation(
            name="sequential",
            build=einspace.build_sequential_module,
            infer=einspace.infer_sequential_module,
            valid=einspace.valid_sequential_module,
            inherit=[einspace.inherit_first_child, einspace.inherit_other_child],
            give_back=[einspace.give_back_default, einspace.give_back_default],
            type="nonterminal",
            child_levels=["module", "module"],
        ),
    )


def _replay_recipe(
    first: _LegacySnapshot,
    second: _LegacySnapshot,
    recipe_json: str,
    root_handle: int,
) -> object:
    """Replay native-selected links and copies without making topology decisions."""
    handles: list[object | None] = [*first.nodes, *second.nodes]

    def get(handle: int) -> object:
        try:
            value = handles[handle]
        except IndexError as error:
            raise RuntimeError("native materialization referenced an unknown handle") from error
        if value is None:
            raise RuntimeError("native materialization used an uninitialized handle")
        return value

    def put(handle: int, value: object) -> None:
        if handle < 0:
            raise RuntimeError("native materialization used a negative handle")
        if handle >= len(handles):
            handles.extend([None] * (handle + 1 - len(handles)))
        handles[handle] = value

    for encoded in json.loads(recipe_json):
        kind, payload = _recipe_payload(encoded)
        if kind == "deep_copy":
            memo: dict[int, object] = {}
            source = get(payload["source"])
            clone = copy.deepcopy(source, memo)
            for old, new in payload["mapping"]:
                original = get(old)
                copied = memo.get(id(original))
                if copied is None:
                    if old != payload["source"]:
                        raise RuntimeError(
                            "native deepcopy recipe omitted a reachable source object"
                        )
                    copied = clone
                put(new, copied)
        elif kind == "new_sequence":
            parent = None if payload["parent"] is None else get(payload["parent"])
            put(payload["node"], _make_sequence(get(payload["template"]), parent, payload["id"]))
        elif kind == "set_children":
            get(payload["node"]).children = [get(child) for child in payload["children"]]
        elif kind == "replace_child":
            get(payload["node"]).children[payload["index"]] = get(payload["child"])
        elif kind == "set_parent":
            get(payload["node"]).parent = (
                None if payload["parent"] is None else get(payload["parent"])
            )
        elif kind == "set_id":
            get(payload["node"]).id = int(payload["id"])
        else:
            raise RuntimeError(f"unknown native materialization recipe event {kind!r}")
    return get(root_handle)


class Alignment:
    """Retained native edit plan plus lazily materialized historical views."""

    def __init__(
        self,
        parent1,
        parent2,
        collapse_corners=False,
        limiter=None,
        profile=False,
        limits=None,
        *,
        workers=1,
    ):
        workers = _core.validate_workers(workers)
        self.profile = profile
        self.limits = limits
        self._materialization_seconds = 0.0
        self.collapse_corners = collapse_corners
        self.verbose = False
        self.model1 = parent1
        self.model2 = parent2
        self.limiter = (
            limiter if limiter is not None and not isinstance(parent1, Architecture) else None
        )
        self._first_legacy: _LegacySnapshot | None = None
        self._second_legacy: _LegacySnapshot | None = None
        if isinstance(parent1, Architecture) or isinstance(parent2, Architecture):
            if not isinstance(parent1, Architecture) or not isinstance(parent2, Architecture):
                raise TypeError("both parents must be portable Architectures or legacy trees")
            _check_compatible_architectures(parent1, parent2)
            first, second = parent1, parent2
        else:
            origins: dict[int, int] = {}
            self._first_legacy = _LegacySnapshot.from_root(parent1, origins)
            self._second_legacy = _LegacySnapshot.from_root(parent2, origins)
            first, second = self._first_legacy.architecture, self._second_legacy.architecture
            if self.limiter is None:
                self.limiter = parent1.limiter
        self._prepared = _core.prepare_architectures(
            first.to_json(),
            second.to_json(),
            profile=profile,
            limits=limits,
            legacy_origins=(
                None
                if self._first_legacy is None
                else (self._first_legacy.origins, self._second_legacy.origins)
            ),
        )
        if self._second_legacy is not None:
            # This effect intentionally occurs before native analysis, including failure paths.
            self._second_legacy.apply_parent2_ids(self._prepared.parent2_ids)
        self._native = self._prepared.analyze(
            _plan_id(), collapse_corners, _memory_check(self.limiter), workers=workers
        )
        self.id = self._native.id
        self.distance = self._native.distance
        self.stats = json.loads(self._native.stats_json())
        self.compute_time = None
        self._paths: list[list[MatrixOperation]] | None = None
        self._selected_path: list[MatrixOperation] | None = None
        self._model_ops1 = None
        self._model_ops2 = None
        self._selected_indices = tuple(self._native.operations)
        self._unordered_indices = tuple(self._native.operations_unordered)
        self._nontrivial_indices = tuple(self._native.nontrivial)
        self._legacy_fingerprint = (
            None
            if self._first_legacy is None
            else (self._first_legacy.fingerprint(), self._second_legacy.fingerprint())
        )

    def __deepcopy__(self, memo):
        self._ensure_fresh()
        result = type(self).__new__(type(self))
        memo[id(self)] = result
        result.__dict__.update(copy.deepcopy(self.__dict__, memo))
        if result._first_legacy is not None:
            result._legacy_fingerprint = (
                result._first_legacy.fingerprint(),
                result._second_legacy.fingerprint(),
            )
        return result

    def _prepared_nodes(self, records, snapshot: _LegacySnapshot | None):
        result = []
        for identifier, name, children, parent_arity, occurrence in records:
            if (
                snapshot is not None
                and occurrence is not None
                and not name.startswith("wrap_")
                and name != "start_node"
            ):
                result.append(snapshot.nodes[occurrence])
            else:
                external = int(identifier) if snapshot is not None else identifier
                prepared = _PreparedNode(external, name, children, parent_arity)
                if snapshot is not None and occurrence is not None:
                    prepared.parent = snapshot.nodes[occurrence]
                result.append(prepared)
        return result

    def breakdown(self, node):
        """Return native boundary-token views without alignment or ID mutation."""
        if isinstance(node, Architecture):
            return self._prepared_nodes(_core.architecture_tokens(node.to_json()), None)[1:]
        snapshot = _LegacySnapshot.from_root(node)
        records = _core.architecture_tokens(snapshot.architecture.to_json())
        return self._prepared_nodes(records, snapshot)[1:]

    @property
    def model_ops1(self):
        if self._model_ops1 is None:
            self._model_ops1 = self._prepared_nodes(
                self._prepared.first_tokens(), self._first_legacy
            )
        return self._model_ops1

    @property
    def model_ops2(self):
        if self._model_ops2 is None:
            self._model_ops2 = self._prepared_nodes(
                self._prepared.second_tokens(), self._second_legacy
            )
        return self._model_ops2

    def _decode_path(self, encoded: list[dict[str, Any]]) -> list[MatrixOperation]:
        def external_id(value):
            return int(value) if value is not None and self._first_legacy is not None else value

        operations = [
            MatrixOperation(
                op_id=item["id"],
                op_type=item["op_type"],
                node1_id=external_id(item["node1_id"]),
                node2_id=external_id(item["node2_id"]),
                i=item["i"],
                j=item["j"],
                ii=item.get("ii"),
                jj=item.get("jj"),
                value=item["value"],
            )
            for item in encoded
        ]
        for operation, item in zip(operations, encoded, strict=True):
            operation.i_swapped = item["i_swapped"]
            operation.j_swapped = item["j_swapped"]
            operation.enabler_ops = [
                operations[value]
                if isinstance(value, int)
                else [operations[index] for index in value]
                for value in item["enabler_ops"]
            ]
            operation.disabler_ops = [
                operations[value]
                if isinstance(value, int)
                else [operations[index] for index in value]
                for value in item["disabler_ops"]
            ]
        return operations

    @property
    def paths(self) -> list[list[MatrixOperation]]:
        if self._paths is None:
            paths = [self._decode_path(path) for path in json.loads(self._native.paths_json())]
            if self._selected_path is None:
                self._selected_path = paths[self._native.path_index]
            else:
                paths[self._native.path_index] = self._selected_path
            self._paths = paths
        return self._paths

    def _current_path(self) -> list[MatrixOperation]:
        if self._selected_path is None:
            if self._paths is not None:
                self._selected_path = self._paths[self._native.path_index]
            else:
                self._selected_path = self._decode_path(
                    json.loads(self._native.selected_path_json())
                )
        return self._selected_path

    @property
    def operations(self) -> list[MatrixOperation]:
        path = self._current_path()
        return [path[index] for index in self._selected_indices]

    @property
    def operations_unordered(self) -> list[MatrixOperation]:
        path = self._current_path()
        return [path[index] for index in self._unordered_indices]

    @property
    def nontrivial_ops(self) -> list[MatrixOperation]:
        path = self._current_path()
        return [path[index] for index in self._nontrivial_indices]

    def _operation_at(self, index: int) -> MatrixOperation:
        return self._current_path()[index]

    def _ensure_fresh(self) -> None:
        if self._legacy_fingerprint is None:
            return
        assert self._first_legacy is not None and self._second_legacy is not None
        current = (self._first_legacy.fingerprint(), self._second_legacy.fingerprint())
        if current != self._legacy_fingerprint:
            raise ValueError("legacy parents changed after this alignment was prepared")

    def select(self, indices_or_mask: Sequence[int] | str) -> Selection:
        self._ensure_fresh()
        if isinstance(indices_or_mask, str):
            if len(indices_or_mask) != len(self._nontrivial_indices) or set(indices_or_mask) - {
                "0",
                "1",
            }:
                raise ValueError("selection mask must match the plan's nontrivial operations")
            indices = tuple(
                index
                for bit, index in zip(indices_or_mask, self._nontrivial_indices, strict=True)
                if bit == "1"
            )
        else:
            indices = tuple(indices_or_mask)
            if any(not isinstance(index, int) or isinstance(index, bool) for index in indices):
                raise TypeError("selection indices must be integers")
        self._native.validate_selection(indices)
        return Selection(self, indices)

    def probabilities(self, skewness=0):
        """Return native enumeration-order masks, costs, and probabilities."""
        self._ensure_fresh()
        return self._native.probabilities(float(skewness), _memory_check(self.limiter))

    def sample(self, skewness=0, *, sampler="native", seed=None, rng=None) -> Selection:
        self._ensure_fresh()
        if sampler == "native":
            from .sampling import resolve_rng

            stream = resolve_rng(sampler, seed, rng)
            mask, cost = self._native.sample(float(skewness), stream, _memory_check(self.limiter))
            return Selection(self, self.select(mask).indices, float(cost))
        if sampler == "reference":
            from .sampling import resolve_rng
            from .sampling_reference import select_mask

            resolve_rng(sampler, seed, rng)
            masks, costs = self._native.combinations(_memory_check(self.limiter))
            return self.select(select_mask(masks, costs, float(skewness)))
        raise ValueError("sampler must be 'native' or 'reference'")

    def _selection_indices(self, selected_ops: object, *, validate: bool) -> tuple[int, ...]:
        if isinstance(selected_ops, Selection):
            if selected_ops._owner is not self or selected_ops.plan_id != self.id:
                raise ValueError("selection belongs to another plan")
            return selected_ops.indices
        if selected_ops is None:
            return self._nontrivial_indices
        if not isinstance(selected_ops, Iterable):
            raise TypeError("selected operations must be a plan Selection or iterable")
        # Compatibility API: map equality-by-logical-ID legacy operation objects to this plan.
        remaining = list(self._selected_indices)
        result: list[int] = []
        for selected in selected_ops:
            try:
                position = next(
                    number
                    for number, index in enumerate(remaining)
                    if self._operation_at(index).id == selected.id
                )
            except (AttributeError, StopIteration) as error:
                raise ValueError("selected operation does not belong to this plan") from error
            result.append(remaining.pop(position))
        return tuple(result)

    @property
    def execution(self) -> dict[str, bool | int | None]:
        """Call-local execution diagnostics, separate from canonical algorithm stats."""
        return json.loads(self._native.execution_json())

    @property
    def timings(self) -> dict[str, float]:
        if not self.profile:
            return {}
        values = json.loads(self._native.profile_json())
        values["materialization"] = self._materialization_seconds
        return values

    def profile_json(self) -> str:
        return json.dumps(self.timings, sort_keys=True, separators=(",", ":"))

    def _apply_selection(self, indices: Sequence[int], *, validate: bool):
        memory_check = _memory_check(self.limiter)
        if self._first_legacy is None:
            architecture_json, recipe_json, root_handle = self._native.apply(
                indices, validate, memory_check
            )
        else:
            architecture_json = None
            recipe_json, root_handle = self._native.apply_legacy(indices, validate, memory_check)
        return self._materialize(architecture_json, recipe_json, root_handle)

    def _materialize(self, architecture_json: str | None, recipe_json: str, root_handle: int):
        if self._first_legacy is None:
            assert architecture_json is not None
            return Architecture._from_validated_json(architecture_json)
        assert self._second_legacy is not None
        started = perf_counter() if self.profile else None
        try:
            return _replay_recipe(self._first_legacy, self._second_legacy, recipe_json, root_handle)
        finally:
            if started is not None:
                self._materialization_seconds += perf_counter() - started

    def generate_offspring(self, selected_ops=None):
        self._ensure_fresh()
        indices = self._selection_indices(selected_ops, validate=False)
        return self._apply_selection(indices, validate=False)


def apply_edits(plan: Alignment, selection: Selection):
    """Apply a validated, plan-bound selection without enumerating alternatives."""
    if not isinstance(plan, Alignment) or not isinstance(selection, Selection):
        raise TypeError("apply_edits requires an Alignment and its Selection")
    if selection._owner is not plan or selection.plan_id != plan.id:
        raise ValueError("selection belongs to another plan")
    plan._ensure_fresh()
    return plan._apply_selection(selection.indices, validate=True)


def _validate_raw_sampler(sampler, seed, rng) -> None:
    if sampler == "native" and seed is None and rng is None:
        return
    from .sampling import resolve_rng

    resolve_rng(sampler, seed, rng)


def raw_crossover(
    parent1,
    parent2,
    skewness=0,
    limiter=None,
    *,
    sampler="native",
    seed=None,
    rng=None,
    profile=False,
    limits=None,
):
    """Return the historical six-item raw result using native selection/application."""
    if _same_raw_parent(parent1, parent2):
        _validate_raw_sampler(sampler, seed, rng)
        return (parent1, [], [], 0, 0, 0)
    matrix = Alignment(parent1, parent2, limiter=limiter, profile=profile, limits=limits)
    operations = matrix.nontrivial_ops
    if not operations:
        _validate_raw_sampler(sampler, seed, rng)
        return (parent1, [], [], 0, 0, 0)
    selection = matrix.sample(skewness, sampler=sampler, seed=seed, rng=rng)
    child = apply_edits(matrix, selection)
    distance_between_parents = matrix.distance
    distance_to_parent2 = selection.cost
    distance_to_parent1 = distance_between_parents - distance_to_parent2
    if isinstance(child, Architecture):
        return (
            child,
            list(selection.operations),
            operations,
            distance_to_parent1,
            distance_to_parent2,
            distance_between_parents,
        )
    # The reference returns one final independent deepcopy of the materialized child.
    child = copy.deepcopy(child)
    return (
        child,
        copy.deepcopy(list(selection.operations)),
        copy.deepcopy(operations),
        distance_to_parent1,
        distance_to_parent2,
        distance_between_parents,
    )
