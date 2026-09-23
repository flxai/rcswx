#!/usr/bin/env python3
"""Validate bundled source/targets/static artifacts; never run neural networks.

Uses only the Python standard library. The checksum file excludes itself.
The companion tests in fixtures/test_recovery.py validate the complete set of
supplied variants; this check additionally enforces the four table pairings.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
from typing import Any

BASE = Path(__file__).resolve().parent
EXPECTED = {
    "resnet18_no_maxpool": 267,
    "resnet34_no_maxpool": 499,
    "mlpmixer_d8": 264,
    "mlpmixer_d12": 392,
}
PAIRS = [
    ("resnet18_no_maxpool", "mlpmixer_d8"),
    ("resnet18_no_maxpool", "mlpmixer_d12"),
    ("resnet34_no_maxpool", "mlpmixer_d8"),
    ("resnet34_no_maxpool", "mlpmixer_d12"),
]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def contained(relative: str) -> Path:
    path = (BASE / relative).resolve()
    require(path.is_relative_to(BASE), f"Path escapes bundle: {relative}")
    require(path.is_file(), f"Missing file: {relative}")
    return path


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_recovery():
    path = BASE / "fixtures" / "recover.py"
    spec = importlib.util.spec_from_file_location("table42_static_recovery", path)
    require(spec is not None and spec.loader is not None, "Cannot load recovery helper")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def validate_grouping_summaries(recovery) -> None:
    """Check the handoff's compact ResNet equations against supplied trees."""
    text = (BASE / "HANDOFF.md").read_text(encoding="utf-8")
    block = next(
        block for block in re.findall(r"```text\n(.*?)\n```", text, re.S)
        if "ResNet18:" in block and "ResNet34:" in block
    )
    values, _, _ = recovery.read_assignments(BASE / "fixtures/source/baselines/resnet.py")

    def sequence(a, b):
        return {"operation": "sequential", "children": [a, b]}

    def residual(width):
        value = values["resnet_block"].render([f"linear{width}"] * 2)
        return recovery.DSLParser(value).parse()

    def downsample(width):
        value = values["resnet_strided_block"].render([f"linear{width}"] * 2)
        return recovery.DSLParser(value).parse()

    stem = recovery.DSLParser(values["resnet_stem_no_maxpool"]).parse()
    functions = {"S": sequence, "R": residual, "D": downsample}

    def evaluate(node):
        if isinstance(node, ast.Constant) and type(node.value) is int:
            return node.value
        if isinstance(node, ast.Name) and node.id == "H":
            return stem
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id in functions and not node.keywords):
            return functions[node.func.id](*[evaluate(arg) for arg in node.args])
        raise ValueError("Unsupported expression in compact grouping summary")

    for label, symbol in (("ResNet18", "resnet18_no_maxpool"),
                          ("ResNet34", "resnet34_no_maxpool")):
        expression = block.split(label + ":", 1)[1]
        if label == "ResNet18":
            expression = expression.split("ResNet34:", 1)[0]
        observed = evaluate(ast.parse(expression.strip(), mode="eval").body)
        expected = read_json(BASE / f"fixtures/trees/{symbol}.json")
        require(observed == expected, f"Handoff grouping mismatch: {symbol}")


def main() -> int:
    checksum_manifest = read_json(BASE / "CHECKSUMS.json")
    for relative, expected in checksum_manifest["files"].items():
        require(digest(contained(relative)) == expected, f"Checksum mismatch: {relative}")
    recovery = load_recovery()
    fixture_manifest = read_json(BASE / "fixtures/manifest.json")
    targets = read_json(BASE / "targets.json")
    require(set(targets["models"]) == set(EXPECTED), "Wrong primary model set")
    require([(p["parent1"], p["parent2"]) for p in targets["ordered_pairs"]] == PAIRS,
            "Wrong table pair order")
    require(targets["primary_collapse_corners"] == [False, True], "Both flags are required")
    last = targets["ordered_pairs"][-1]
    require(last["printed_compute_time"] == "741.57 minutes", "Historical unit changed")
    require(last["literal_time_seconds"] == 44494.2, "Incorrect literal conversion")
    for symbol, count in EXPECTED.items():
        model = targets["models"][symbol]
        source = contained(model["source_file"])
        values, _, _ = recovery.read_assignments(source)
        definition = values[symbol]
        raw = contained(model["definition_path"])
        require(raw.read_bytes() == definition.encode("utf-8"), f"Expansion mismatch: {symbol}")
        require(digest(source) == model["source_file_sha256"], f"Wrong source: {symbol}")
        require(digest(raw) == model["definition_sha256"], f"Wrong descriptor: {symbol}")
        tree = recovery.DSLParser(definition).parse()
        recovery.check_arities(tree)
        require(tree == read_json(contained(model["static_tree_path"])), f"Tree mismatch: {symbol}")
        stream = recovery.loader_tokens(definition)
        require(len(stream) == count == model["static_operation_occurrences"],
                f"Incorrect static count: {symbol}")
        require(len(list(recovery.preorder(tree))) == count, f"Preorder count: {symbol}")
        require(fixture_manifest["models"][symbol]["expanded_operation_count"] == count,
                f"Fixture manifest mismatch: {symbol}")
        print(f"PASS {symbol}: {count} static operation occurrences")
    validate_grouping_summaries(recovery)
    print(f"PASS {len(checksum_manifest['files'])} bundled file checksums")
    print("PASS four ordered pairs, both pruning modes, original units, compact ResNet groupings")
    print("SCOPE: static source/descriptor validation only; no model execution or benchmark")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError, StopIteration, SyntaxError) as error:
        print(f"VALIDATION FAILED: {error}", file=sys.stderr)
        raise SystemExit(1) from error
