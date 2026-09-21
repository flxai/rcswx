"""Source-derived production inventory and actual construction cases.

Run as ``python -m tests.reference.corpus OUTPUT.json`` in the reference runtime.
The generated inventory records source rows, environment, and reference outcomes;
expected failures remain rows rather than being removed from coverage.
"""

import copy
import functools
import inspect
import json
import math
import sys
from pathlib import Path

import torch

from .original import environment, limits, load


def callback_record(callback):
    if isinstance(callback, functools.partial):
        return {
            "function": callback.func.__name__,
            "args": callback.args,
            "kwargs": callback.keywords,
        }
    return {"function": callback.__name__}


def registries(module):
    return {
        name: value
        for name, value in vars(module).items()
        if type(value) is dict and "network" in value
    }


def inventory():
    reference = load()
    rows = []
    for family in ("einspace", "hnasbench201"):
        module = getattr(reference, family)
        for registry_name, grammar in registries(module).items():
            for level, rules in grammar.items():
                for index, (operation, probability) in enumerate(
                    zip(rules["options"], rules["probs"], strict=True)
                ):
                    rows.append(
                        {
                            "family": family,
                            "registry": registry_name,
                            "level": level,
                            "index": index,
                            "probability": probability,
                            "name": operation.name,
                            "type": operation.type,
                            "child_levels": operation.child_levels,
                            "build": callback_record(operation.build),
                            "infer": callback_record(operation.infer),
                            "valid": callback_record(operation.valid),
                            "inherit": [callback_record(fn) for fn in operation.inherit],
                            "give_back": [callback_record(fn) for fn in operation.give_back],
                        }
                    )
    return rows


def operation_map(module):
    result = {}
    for grammar in registries(module).values():
        for rules in grammar.values():
            for operation in rules["options"]:
                result.setdefault(operation.name, operation)
    return result


def computation(name="identity"):
    return ("computation", name)


def branch(arity, inner=None, split=None, merge=None):
    inner = inner or computation()
    return (
        f"branching({arity})",
        split or f"clone({arity})",
        inner,
        *([inner] if arity == 2 else []),
        merge or f"add({arity})",
    )


def construction_cases():
    reference = load()
    cases = {}
    image = (2, 8, 16, 16)
    for operation in operation_map(reference.einspace).values():
        name = operation.name
        if name == "sequential":
            description = (name, computation("linear(16)"), computation("relu"))
        elif name.startswith("sequential("):
            description = (name, computation("linear(16)"))
        elif name.startswith("branching("):
            description = branch(int(name[10:-1]), computation("linear(16)"))
        elif name == "routing":
            description = (name, "im2col(3,1,1)", computation(), "col2im")
        elif name == "computation":
            description = computation("linear(16)")
        elif name.startswith(("clone(", "group(")):
            arity = int(name.split("(")[1].split(",")[0].rstrip(")"))
            description = branch(arity, split=name)
        elif name.startswith(("add(", "cat(")):
            arity = int(name.split("(")[1].split(",")[0].rstrip(")"))
            description = branch(arity, merge=name)
        elif name.startswith("dot_product"):
            description = branch(2, merge=name)
        elif name.startswith("broadcast("):
            description = (
                "branching(2)",
                "clone(2)",
                computation("linear(16)"),
                computation("linear(32)"),
                name,
            )
        elif name.startswith("perm("):
            shape = (2, 8, 8) if name.count(",") == 2 else image
            cases[f"einspace/{name}/pre"] = (
                "einspace",
                ("routing", name, computation(), "identity"),
                shape,
                "module",
            )
            cases[f"einspace/{name}/post"] = (
                "einspace",
                ("routing", "identity", computation(), name),
                shape,
                "module",
            )
            continue
        elif name.startswith("im2col("):
            description = ("routing", name, computation(), "col2im")
        elif name == "col2im":
            description = ("routing", "im2col(1,1,0)", computation(), name)
        else:
            description = computation(name)
        cases[f"einspace/{name}"] = ("einspace", description, image, "module")
    for rank, shape in ((2, (2, 8)), (3, (2, 8, 8)), (4, image)):
        for name in ("identity", "norm", "relu", "softmax", "pos_enc", "linear(16)"):
            cases[f"einspace/{name}/rank{rank}"] = ("einspace", computation(name), shape, "module")
    for factor in (0.5, 2):
        cases[f"einspace/linear(x{factor})"] = (
            "einspace",
            computation(f"linear(x{factor})"),
            (2, 8, 8),
            "module",
        )
        cases[f"einspace/repeated-relative-{factor}"] = (
            "einspace",
            ("sequential(4)", computation(f"linear(x{factor})")),
            (2, 8, 32),
            "module",
        )

    module = reference.hnasbench201
    grammar = module.grammar
    defaults = {level: rules["options"][0] for level, rules in grammar.items()}
    defaults["OP"] = next(op for op in grammar["OP"]["options"] if op.name == "identity")

    def expand(operation):
        if not operation.child_levels:
            return operation.name
        return (operation.name, *(expand(defaults[level]) for level in operation.child_levels))

    for level, rules in grammar.items():
        for operation in rules["options"]:
            cases[f"hnasbench201/{operation.name}"] = (
                "hnasbench201",
                expand(operation),
                image,
                level,
            )
    return cases


def input_params(shape):
    return {
        "shape": torch.Size(shape),
        "other_shape": None,
        "mode": "im" if len(shape) == 4 else "col",
        "other_mode": None,
        "branching_factor": 1,
        "last_im_shape": None,
    }


def build_case(case, *, owned=False, build_model=True):
    family, description, shape, level = case
    reference = load()
    if owned:
        from rcswx import genotype, pcfg, reconstruction
        from rcswx.grammars import einspace, hnasbench201

        module = {"einspace": einspace, "hnasbench201": hnasbench201}[family]
        node_type, pcfg_type, sampler_type = (
            genotype.DerivationTreeNode,
            pcfg.PCFG,
            reconstruction.Reconstructor,
        )
    else:
        module = getattr(reference, family)
        node_type, pcfg_type, sampler_type = (
            reference.state.DerivationTreeNode,
            reference.pcfg.PCFG,
            reference.sampler.Sampler,
        )
    operations = operation_map(module)
    grammar = copy.deepcopy(
        module.deep_broadcast_grammar if family == "einspace" else module.grammar
    )
    if family == "einspace":
        for factor in (0.5, 2):
            operation = module.linear_x(factor)
            operations[operation.name] = operation
            grammar["computation_fn"]["options"].append(operation)
            grammar["computation_fn"]["probs"].append(0.077)
    ordered = []

    def collect(item):
        if isinstance(item, str):
            ordered.append(operations[item])
        else:
            ordered.append(operations[item[0]])
            for child in item[1:]:
                collect(child)

    collect(description)
    guard = limits(batch=torch.zeros(shape))
    sampler = sampler_type(pcfg_type(grammar, guard), guard, "iterative")
    root = node_type(1, level=level, input_params=input_params(shape), limiter=guard)
    root = sampler.sample(root.input_params, operations=ordered, root=root)
    return root, root.build(root) if build_model else None


def reference_outcomes():
    outcomes = []
    reference = load()
    for name, case in construction_cases().items():
        torch.manual_seed(123)
        row = {
            "case": name,
            "family": case[0],
            "description": case[1],
            "input_shape": case[2],
            "root_level": case[3],
        }
        try:
            root, model = build_case(case)
            batch = torch.linspace(-1, 1, steps=math.prod(case[2])).reshape(case[2])
            result = model(batch)
            row.update(
                status="success",
                output_shape=list(result.shape),
                parameters=sum(p.numel() for p in model.parameters()),
            )
            try:
                reference.algorithm.AlignmentMatrixRecursive(
                    copy.deepcopy(root), copy.deepcopy(root), limiter=limits()
                )
            except Exception as error:
                row["alignment"] = {
                    "status": "reference_failure",
                    "exception": type(error).__name__,
                }
            else:
                row["alignment"] = {"status": "success"}
        except Exception as error:
            row.update(status="reference_failure", exception=type(error).__name__)
        outcomes.append(row)
    return outcomes


def main():
    destination = Path(sys.argv[1])
    torch.set_num_threads(1)
    reference = load()
    factories = {
        family: {
            name: str(inspect.signature(value))
            for name, value in vars(getattr(reference, family)).items()
            if inspect.isfunction(value)
            and name
            in (
                "linear",
                "linear_x",
                "permute",
                "im2col",
                "clone",
                "group",
                "cat",
                "add",
                "dot_product",
                "broadcast",
            )
        }
        for family in ("einspace", "hnasbench201")
    }
    payload = {
        "environment": environment(),
        "productions": inventory(),
        "factories": factories,
        "cases": reference_outcomes(),
    }
    destination.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        json.dumps(
            {
                "inventory": str(destination),
                "production_rows": len(payload["productions"]),
                "construction_cases": len(payload["cases"]),
            }
        )
    )


if __name__ == "__main__":
    main()
