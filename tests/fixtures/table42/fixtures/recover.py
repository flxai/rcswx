#!/usr/bin/env python3
"""Recover coauthor baseline strings without importing einsearch or executing models.

Only a small, explicitly supported subset of Python expressions is interpreted.
The source files are preserved verbatim; no source or grammar repairs are made.
The exported trees are syntax trees, not runtime DerivationTreeNode instances.
"""
from __future__ import annotations

import ast
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

BASE = Path(__file__).resolve().parent
SOURCES = BASE / "source" / "baselines"
MODELS = {
    "resnet.py": ("resnet18_no_maxpool", "resnet18_conv7x7_no_maxpool", "resnet34_no_maxpool"),
    "mlpmixer.py": ("mlpmixer_d2", "mlpmixer_d4", "mlpmixer_d8", "mlpmixer_d12"),
    "vit.py": ("vit_d2", "vit_d4", "vit_d8"),
    "wideresnet.py": ("wideresnet16_4",),
    "convnextv2.py": ("convnextv2_tiny",),
}
TARGETS = {
    "resnet18_no_maxpool": 267,
    "resnet34_no_maxpool": 499,
    "mlpmixer_d8": 264,
    "mlpmixer_d12": 392,
}
ORIGINAL_NAMES = {
    "resnet.py": "resnet(1).py", "mlpmixer.py": "mlpmixer(1).py",
    "__init__.py": "__init__(1).py", "vit.py": "vit.py",
    "wideresnet.py": "wideresnet.py", "convnextv2.py": "convnextv2.py",
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


class Template:
    def __init__(self, expression: ast.Lambda, environment: dict[str, Any]):
        args = expression.args
        if args.posonlyargs or args.vararg or args.kwonlyargs or args.kwarg or args.defaults:
            raise ValueError("Unsupported template signature")
        self.parameters = [arg.arg for arg in args.args]
        self.expression = expression.body
        self.environment = environment

    def render(self, arguments: list[Any]) -> str:
        if len(arguments) != len(self.parameters):
            raise ValueError("Wrong number of template arguments")
        return evaluate(self.expression, self.environment | dict(zip(self.parameters, arguments)))


def evaluate(node: ast.AST, environment: dict[str, Any]) -> Any:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        return environment[node.id]
    if isinstance(node, ast.Lambda):
        return Template(node, environment)
    if isinstance(node, ast.JoinedStr):
        return "".join(evaluate(part, environment) for part in node.values)
    if isinstance(node, ast.FormattedValue):
        if node.conversion != -1 or node.format_spec is not None:
            raise ValueError("Unsupported f-string formatting")
        value = evaluate(node.value, environment)
        if not isinstance(value, str):
            raise TypeError("Expected a string interpolation")
        return value
    if isinstance(node, ast.Call) and not node.keywords:
        function = evaluate(node.func, environment)
        if not isinstance(function, Template):
            raise TypeError("Only the source's simple templates may be called")
        return function.render([evaluate(arg, environment) for arg in node.args])
    raise ValueError(f"Unsupported source expression: {ast.dump(node)}")


def read_assignments(path: Path):
    environment: dict[str, Any] = {}
    ranges: dict[str, list[int]] = {}
    duplicates = []
    for node in ast.parse(path.read_text(), filename=str(path)).body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            raise ValueError(f"Expected a single-name assignment in {path}:{node.lineno}")
        name = node.targets[0].id
        if name in environment:
            duplicates.append({"symbol": name, "previous_lines": ranges[name],
                               "replacement_lines": [node.lineno, node.end_lineno]})
        environment[name] = evaluate(node.value, environment)
        ranges[name] = [node.lineno, node.end_lineno]
    return environment, ranges, duplicates


def loader_tokens(text: str) -> list[str]:
    """Mirror the uploaded build_baseline's token splitting, without eval()."""
    tokens = [part.strip() for part in re.split(r",|\[|\]", re.sub(r"\s+", "", text)) if part.strip()]
    remove = []
    for a, b in zip(range(len(tokens) - 1), range(1, len(tokens))):
        if "cat" in tokens[a]:
            tokens[a] = f"{tokens[a]}, {tokens[b]}"
            remove.append(b)
    return [token for i, token in enumerate(tokens) if i not in remove]


class DSLParser:
    """Preserve explicit brackets and operation names; tolerate source commas.

    The source loader discards commas and brackets. Some supplied strings omit a
    comma between children or have trailing commas. Do not repair their bytes.
    """
    def __init__(self, text: str):
        self.text = re.sub(r"\s+", "", text)
        self.position = 0

    def node(self) -> dict[str, Any]:
        start = self.position
        nesting = 0
        while self.position < len(self.text):
            char = self.text[self.position]
            if nesting == 0 and char in "[],":
                break
            if char == "(":
                nesting += 1
            elif char == ")":
                nesting -= 1
                if nesting < 0:
                    raise ValueError("Unbalanced operation parentheses")
            self.position += 1
        if nesting or self.position == start:
            raise ValueError(f"Invalid operation at offset {start}")
        result = {"operation": self.text[start:self.position], "children": []}
        if self.position < len(self.text) and self.text[self.position] == "[":
            self.position += 1
            while self.position < len(self.text) and self.text[self.position] != "]":
                if self.text[self.position] == ",":
                    self.position += 1
                else:
                    result["children"].append(self.node())
            if self.position >= len(self.text):
                raise ValueError("Unclosed child list")
            self.position += 1
        return result

    def parse(self):
        result = self.node()
        if self.position != len(self.text):
            raise ValueError(f"Unexpected trailing content at {self.position}")
        return result


def preorder(node):
    yield node["operation"]
    for child in node["children"]:
        yield from preorder(child)


def check_arities(node):
    expected = {"sequential": 2, "routing": 3, "computation": 1,
                "branching(2)": 4, "branching(4)": 3, "branching(8)": 3}
    count = expected.get(node["operation"], 0)
    if len(node["children"]) != count:
        raise ValueError(f"Unexpected child count for {node['operation']}: {len(node['children'])}, expected {count}")
    for child in node["children"]:
        check_arities(child)


def load_aliases():
    module = ast.parse((SOURCES / "__init__.py").read_text())
    function = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "build_baseline")
    assignment = next(node for node in function.body if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == "op_map" for target in node.targets))
    return ast.literal_eval(assignment.value)


def dump_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def main():
    for directory in ("expanded", "trees", "operations"):
        (BASE / directory).mkdir(exist_ok=True)
    aliases = load_aliases()
    definitions = {}
    manifest = {
        "schema": 1,
        "scope": "Static, source-derived architecture recovery. No einsearch imports, tensor execution, or alignment benchmark.",
        "provenance": "User-uploaded coauthor files, identified by the coauthor in the conversation as the definitions used; no source commit supplied.",
        "count_convention": "One per operation occurrence in the uploaded loader's flattened DSL stream. Not RCSWX alignment-token count, expanded repetition count, parameter count, or runtime-verified serialise() length.",
        "sources": {}, "models": {},
        "table_pairs": [
            ["resnet18_no_maxpool", "mlpmixer_d8"],
            ["resnet18_no_maxpool", "mlpmixer_d12"],
            ["resnet34_no_maxpool", "mlpmixer_d8"],
            ["resnet34_no_maxpool", "mlpmixer_d12"],
        ],
        "runtime_model_build_validated": False,
        "benchmark_executed": False,
        "historical_collapse_corners": None,
        "historical_input_params": None,
        "historical_source_commit": None,
    }
    for path in sorted(SOURCES.glob("*.py")):
        manifest["sources"][path.name] = {
            "upload_name": ORIGINAL_NAMES[path.name],
            "relative_path": str(path.relative_to(BASE)),
            "sha256": sha256(path.read_bytes()),
        }
    for filename, names in MODELS.items():
        values, ranges, duplicates = read_assignments(SOURCES / filename)
        manifest["sources"][filename]["reassigned_symbols"] = duplicates
        for name in names:
            definition = values[name]
            if not isinstance(definition, str):
                raise TypeError(f"{name} is not a string")
            tokens = loader_tokens(definition)
            tree = DSLParser(definition).parse()
            check_arities(tree)
            if [re.sub(r"\s+", "", token) for token in tokens] != list(preorder(tree)):
                raise ValueError(f"Loader stream and explicit tree differ for {name}")
            if name in TARGETS and len(tokens) != TARGETS[name]:
                raise ValueError(f"Historical count mismatch for {name}: {len(tokens)}")
            expressions = [aliases.get(token, token) for token in tokens]
            raw_path = BASE / "expanded" / f"{name}.einspace.txt"
            raw_path.write_bytes(definition.encode())
            dump_json(BASE / "trees" / f"{name}.json", tree)
            dump_json(BASE / "operations" / f"{name}.json", {
                "source_expressions": tokens,
                "mapped_factory_expressions": expressions,
                "warning": "Factory expressions are unevaluated source strings, not runtime operation.name values.",
            })
            manifest["models"][name] = {
                "source_file": f"source/baselines/{filename}",
                "assignment_lines": ranges[name],
                "expanded_definition": str(raw_path.relative_to(BASE)),
                "expanded_definition_sha256": sha256(definition.encode()),
                "ordered_source_operations_sha256": sha256(canonical(tokens)),
                "syntax_tree_sha256": sha256(canonical(tree)),
                "expanded_operation_count": len(tokens),
                "operation_histogram": dict(sorted(Counter(tokens).items())),
                "historical_count_target": TARGETS.get(name),
                "historical_count_matches": len(tokens) == TARGETS[name] if name in TARGETS else None,
                "static_tree_arity_validated": True,
            }
            definitions[name] = definition
    output = ['"""Expanded coauthor model-definition strings. Generated by recover.py."""', ""]
    for name, definition in definitions.items():
        output.append(f"{name} = {definition!r}\n")
    output.append("MODEL_DEFINITIONS = {\n" + "".join(f"    {name!r}: {name},\n" for name in definitions) + "}\n")
    (BASE / "definitions.py").write_text("\n".join(output))
    dump_json(BASE / "manifest.json", manifest)
    for name, record in manifest["models"].items():
        print(f"{name:32s} {record['expanded_operation_count']:4d}")


if __name__ == "__main__":
    main()
