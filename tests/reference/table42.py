"""Hash-pinned Table 4.2 fixtures; diagnostic configuration is not historical evidence."""

import argparse
import ast
import copy
import functools
import hashlib
import json
import math
import os
import random
import re
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
from types import SimpleNamespace

BUNDLE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "table42"
CHECKSUMS_SHA256 = "468c740edf07e7f04148093b2dbe1973c2ec936bfb0b4f009f1e7c427fe04ed1"
EXTRA_FACTORIES = ("linear_x(4)", "linear_x(0.25)")
INPUT_PARAMS = {
    "shape": [2, 3, 32, 32],
    "other_shape": None,
    "mode": "im",
    "other_mode": None,
    "branching_factor": 1,
    "last_im_shape": None,
}
LIMITS = {
    "time": 120,
    "restart_time": 300,
    "max_id": 10000,
    "depth": 128,
    "memory": 65536,
    "memory_crossover": 65536,
    "individual_memory": 16384,
    "batch_pass_seconds": 120,
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def fingerprint(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def verify_assets(root=BUNDLE_ROOT):
    root = Path(root).resolve()
    if sha256(root / "CHECKSUMS.json") != CHECKSUMS_SHA256:
        raise ValueError("Table 4.2 checksum manifest differs from the supplied bundle")
    manifest = json.loads((root / "CHECKSUMS.json").read_text())
    for relative, expected in manifest["files"].items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or sha256(path) != expected:
            raise ValueError(f"Table 4.2 asset differs from supplied bytes: {relative}")
    return json.loads((root / "targets.json").read_text())


@functools.lru_cache(maxsize=1)
def assets():
    return verify_assets()


def backend(owned):
    from .original import load

    reference = load()
    if not owned:
        return SimpleNamespace(
            grammar=reference.einspace,
            node=reference.state.DerivationTreeNode,
            operation=reference.state.Operation,
            pcfg=reference.pcfg.PCFG,
            sampler=reference.sampler.Sampler,
            limiter=reference.utility.Limiter,
            alignment=reference.algorithm.AlignmentMatrixRecursive,
        )
    from rcswx import PCFG, Alignment, Limiter, Reconstructor
    from rcswx.genotype import DerivationTreeNode, Operation
    from rcswx.grammars import einspace

    return SimpleNamespace(
        grammar=einspace,
        node=DerivationTreeNode,
        operation=Operation,
        pcfg=PCFG,
        sampler=Reconstructor,
        limiter=Limiter,
        alignment=Alignment,
    )


def source_identity():
    from rcswx import _core
    from rcswx.grammars import einspace

    from .original import SOURCE_HASHES, environment, load

    load()
    provenance_path = Path(__file__).resolve().parents[2] / "source-provenance.json"
    provenance = json.loads(provenance_path.read_text()) if provenance_path.exists() else None
    if provenance is not None:
        root = provenance_path.parent
        for relative, expected in provenance["files_sha256"].items():
            path = (root / relative).resolve()
            if not path.is_relative_to(root) or sha256(path) != expected:
                raise ValueError(f"Recorded source snapshot changed: {relative}")
    return {
        "fixture_manifest_sha256": sha256(BUNDLE_ROOT / "fixtures/manifest.json"),
        "loader_sha256": sha256(BUNDLE_ROOT / "fixtures/source/baselines/__init__.py"),
        "parser_sha256": sha256(BUNDLE_ROOT / "fixtures/recover.py"),
        "grammar_source_hashes": {
            "reference": SOURCE_HASHES["grammars/einspace.py"],
            "native": sha256(einspace.__file__),
        },
        "native_extension_sha256": sha256(_core.__file__),
        "reference_environment": environment(),
        "checkout_provenance": provenance,
    }


def resolve_runtime_config(cpu, admit_supplied_factories=False):
    import torch

    assets()
    torch.set_num_threads(1)
    config = {
        "schema": "rcswx-table42-runtime-v1",
        "profile_id": "diagnostic_cpu32_supplied_factories_v1"
        if admit_supplied_factories
        else "diagnostic_cpu32_v1",
        "stratum": "paired_current_reference",
        "historical": False,
        "input_params": copy.deepcopy(INPUT_PARAMS),
        "device": "cpu",
        "dtype": "float32",
        "seed": 42,
        "threads": 1,
        "affinity_cpu": cpu,
        "baseline_grammar": "grammar",
        "replay_grammar": "grammar",
        "extra_factory_expressions": list(EXTRA_FACTORIES) if admit_supplied_factories else [],
        "grammar_admission_note": (
            "Benchmark-local copies admit exactly the supplied relative factories; probabilities "
            "are unused because random fallback is prohibited. Original grammar bytes are unchanged."
            if admit_supplied_factories
            else "Unmodified ordinary grammar; unsupported supplied operations remain failures."
        ),
        "construction_limits": dict(LIMITS),
        "alignment_limits": dict(LIMITS, time=3600, restart_time=3600),
        "call_boundary": "complete_alignment_object_return",
        "construction_seed": 42,
        "call_seed": 42,
        **source_identity(),
    }
    validate_runtime_config(config)
    return config


def validate_runtime_config(config):
    if config.get("schema") != "rcswx-table42-runtime-v1":
        raise ValueError("Expected resolved rc swx Table 4.2 runtime schema, not campaign policy")
    if config.get("stratum") != "paired_current_reference" or config.get("historical") is not False:
        raise ValueError("Only the explicitly nonhistorical paired-current stratum is configured")
    if config.get("input_params") != INPUT_PARAMS:
        raise ValueError("This diagnostic profile requires the explicit (2,3,32,32) input metadata")
    for key, expected in (("device", "cpu"), ("dtype", "float32"), ("threads", 1)):
        if config.get(key) != expected:
            raise ValueError(f"Unexpected diagnostic {key}: {config.get(key)!r}")
    for key in ("seed", "construction_seed", "call_seed"):
        if type(config.get(key)) is not int or config[key] != 42:
            raise ValueError(f"Diagnostic {key} must be explicitly 42")
    cpu = config.get("affinity_cpu")
    if type(cpu) is not int or cpu < 0 or cpu not in os.sched_getaffinity(0):
        raise ValueError(f"Configured CPU is not available to this process: {cpu!r}")
    extras = config.get("extra_factory_expressions")
    if extras not in ([], list(EXTRA_FACTORIES)):
        raise ValueError("Only the two exact supplied Mixer factories can be explicitly admitted")
    expected_profile = "diagnostic_cpu32_supplied_factories_v1" if extras else "diagnostic_cpu32_v1"
    if config.get("profile_id") != expected_profile:
        raise ValueError("Grammar admission changes require a separately named diagnostic profile")
    for key in ("baseline_grammar", "replay_grammar"):
        if config.get(key) != "grammar":
            raise ValueError(f"Unsupported diagnostic grammar selection: {config.get(key)!r}")
    for key in ("construction_limits", "alignment_limits"):
        limits = config.get(key, {})
        if set(limits) != set(LIMITS):
            raise ValueError(f"Explicit {key} must include exactly {sorted(LIMITS)}")
        if any(
            type(value) not in (int, float) or not math.isfinite(value) or value <= 0
            for value in limits.values()
        ):
            raise ValueError(f"All {key} values must be finite and positive")
    if config.get("call_boundary") != "complete_alignment_object_return":
        raise ValueError("The primary boundary must retain the complete returned alignment")
    actual = source_identity()
    for key, value in actual.items():
        if config.get(key) != value:
            raise ValueError(f"Resolved runtime identity changed: {key}")
    return config


def load_runtime_config(path):
    import torch

    torch.set_num_threads(1)
    assets()
    return validate_runtime_config(json.loads(Path(path).read_text()))


def configure_runtime(config):
    import numpy as np
    import torch

    os.sched_setaffinity(0, {config["affinity_cpu"]})
    torch.set_num_threads(config["threads"])
    torch.set_default_dtype(torch.float32)
    random.seed(config["construction_seed"])
    np.random.seed(config["construction_seed"])
    torch.manual_seed(config["construction_seed"])


def factory_expression(expression, module):
    node = ast.parse(expression, mode="eval").body
    if isinstance(node, ast.Name) and not node.id.startswith("_"):
        return getattr(module, node.id)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
        if node.func.id.startswith("_"):
            raise ValueError("Private factory expressions are not accepted")
        return getattr(module, node.func.id)(*[ast.literal_eval(arg) for arg in node.args])
    raise ValueError(f"Unsupported supplied factory expression: {expression}")


def grammar_for(types, name, extras):
    grammar = copy.deepcopy(getattr(types.grammar, name))
    for expression in extras:
        operation = factory_expression(expression, types.grammar)
        options = grammar["computation_fn"]["options"]
        if operation not in options:
            options.append(operation)
            grammar["computation_fn"]["probs"].append(1.0)
    return grammar


def operation_descriptor(operation):
    from .corpus import callback_record

    return {
        "name": operation.name,
        "type": operation.type,
        "child_levels": list(operation.child_levels),
        "build": callback_record(operation.build),
        "infer": callback_record(operation.infer),
        "valid": callback_record(operation.valid),
        "inherit": [callback_record(callback) for callback in operation.inherit],
        "give_back": [callback_record(callback) for callback in operation.give_back],
    }


def runtime_tree_record(root):
    # Reuse the existing alias-sensitive comparison rather than invent a second
    # lossy name/count serializer. The pinned reference environment includes pytest.
    from tests.test_reference_construction import metadata

    return json.loads(canonical(metadata(root)))


def forbid_random_fallback(pcfg):
    def reject(*args, **kwargs):
        raise RuntimeError("Random PCFG fallback during an exact Table 4.2 replay")

    previous = pcfg.sample
    pcfg.sample = reject
    return previous


def supplied_loader(types, config):
    import numpy as np

    path = BUNDLE_ROOT / "fixtures/source/baselines/__init__.py"
    definition = next(
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == "build_baseline"
    )
    namespace = dict(vars(types.grammar))
    guarded = []

    def guarded_pcfg(grammar, limiter):
        pcfg = types.pcfg(grammar, limiter)
        guarded.append((pcfg, forbid_random_fallback(pcfg)))
        return pcfg

    namespace.update(
        Sampler=types.sampler,
        PCFG=guarded_pcfg,
        Limiter=types.limiter,
        grammar=grammar_for(types, config["baseline_grammar"], config["extra_factory_expressions"]),
        re=re,
        np=np,
    )
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["build_baseline"], guarded


def restore_fallbacks(guarded):
    for pcfg, previous in guarded:
        pcfg.sample = previous


def model_root(symbol, owned, config):
    import torch

    target = assets()["models"][symbol]
    types = backend(owned)
    stream = json.loads((BUNDLE_ROOT / target["source_operations_path"]).read_text())
    source = stream["source_expressions"]
    factories = stream["mapped_factory_expressions"]
    resolved = [factory_expression(expression, types.grammar) for expression in factories]
    expected = target["runtime_serialise_length_target"]
    if len(source) != expected or len(factories) != expected:
        raise ValueError(f"Source stream length differs for {symbol}")
    inputs = copy.deepcopy(config["input_params"])
    inputs["shape"] = torch.Size(inputs["shape"])
    loader, guarded = supplied_loader(types, config)
    try:
        baseline = loader(
            (BUNDLE_ROOT / target["definition_path"]).read_text(),
            copy.deepcopy(inputs),
            limits=copy.deepcopy(config["construction_limits"]),
        )
    finally:
        restore_fallbacks(guarded)
    before = list(baseline.serialise())
    if len(before) != expected or len({id(node) for node in before}) != expected:
        raise ValueError(f"Baseline count or unique occurrence identity differs for {symbol}")
    for node, operation in zip(before, resolved, strict=True):
        if canonical(operation_descriptor(node.operation)) != canonical(
            operation_descriptor(operation)
        ):
            raise ValueError(
                f"Source/factory/runtime operation differs at node {node.id} of {symbol}"
            )
    static_tree = json.loads((BUNDLE_ROOT / target["static_tree_path"]).read_text())
    iterator = iter(resolved)

    def check_topology(static, node):
        operation = next(iterator)
        if node.operation.name != operation.name or len(node.children) != len(static["children"]):
            raise ValueError(f"Supplied grouping differs at node {node.id} of {symbol}")
        for static_child, child in zip(static["children"], node.children, strict=True):
            check_topology(static_child, child)

    check_topology(static_tree, baseline)
    if next(iterator, None) is not None:
        raise ValueError("Unconsumed source operation during topology audit")
    guard = types.limiter(
        copy.deepcopy(config["construction_limits"]), batch=torch.zeros(inputs["shape"])
    )
    pcfg = types.pcfg(
        grammar_for(types, config["replay_grammar"], config["extra_factory_expressions"]), guard
    )
    builder = types.sampler(pcfg, guard, "iterative")
    replay = [node.operation for node in before]
    previous = forbid_random_fallback(pcfg)
    try:
        guard.timer.start()
        root = builder.sample(copy.deepcopy(inputs), replay)
    finally:
        pcfg.sample = previous
    after = list(root.serialise())
    if replay or len(after) != expected or len({id(node) for node in after}) != expected:
        raise ValueError(f"Incomplete or nonunique operation replay for {symbol}")
    before_record = runtime_tree_record(baseline)
    after_record = runtime_tree_record(root)
    if before_record != after_record:
        raise ValueError(f"Baseline and replayed topology/metadata differ for {symbol}")
    if any(
        type(node) is not types.node or type(node.operation) is not types.operation
        for node in after
    ):
        raise TypeError(f"Mixed backend ownership in {symbol}")
    tokeniser = types.alignment.__new__(types.alignment)
    tokens = tokeniser.breakdown(root)
    manifest = {
        "symbol": symbol,
        "backend": "native" if owned else "reference",
        "source_sha256": target["source_file_sha256"],
        "definition_sha256": target["definition_sha256"],
        "static_operation_count": len(source),
        "runtime_node_count": len(after),
        "alignment_tokens_including_start": len(tokens) + 1,
        "runtime_sha256": fingerprint(after_record),
        "runtime": after_record,
        "baseline_runtime_sha256": fingerprint(before_record),
        "baseline_replay_equal": True,
        "random_fallback_calls": 0,
        "complete_replay": True,
        "ownership": {
            "node_class": types.node.__module__ + "." + types.node.__name__,
            "operation_class": types.operation.__module__ + "." + types.operation.__name__,
        },
        "operation_mapping": [
            {
                "source": expression,
                "factory": factory,
                "runtime": operation_descriptor(node.operation),
            }
            for expression, factory, node in zip(source, factories, after, strict=True)
        ],
    }
    return root, builder, manifest


def table_pairs(config):
    from .networks import NetworkPair

    return {
        row["pair_id"]: NetworkPair(
            row["pair_id"],
            "einspace",
            "table42",
            max(row["printed_node_counts"]),
            tuple(config["input_params"]["shape"]),
            row["parent1"],
            row["parent2"],
        )
        for row in assets()["ordered_pairs"]
    }


def prepare_pair(case, owned, config):
    import torch

    left, builder, first = model_root(case.left, owned, config)
    right, _, second = model_root(case.right, owned, config)
    types = backend(owned)
    guard = types.limiter(copy.deepcopy(config["alignment_limits"]), batch=torch.zeros(case.shape))
    for root in (left, right):
        for node in root.serialise():
            node.limiter = guard
    builder.limiter = builder.pcfg.limiter = guard
    manifest = {
        "schema": "rcswx-table42-runtime-fixtures-v1",
        "profile_id": config["profile_id"],
        "parents": [first, second],
        "runtime_sha256": fingerprint([first["runtime_sha256"], second["runtime_sha256"]]),
        "fixture_manifest_sha256": config["fixture_manifest_sha256"],
        "loader_sha256": config["loader_sha256"],
        "parser_sha256": config["parser_sha256"],
    }
    return (left, right), builder, guard, manifest


def tensor_record(tensor):
    value = tensor.detach().contiguous()
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "bytes": value.numel() * value.element_size(),
        "sha256": hashlib.sha256(memoryview(value.numpy())).hexdigest(),
    }


def audit_worker(args):
    import numpy as np
    import torch

    from benchmarks.memory import rng_record

    stage = "configuration"
    report = {"model": args.model, "backend": args.backend, "status": "not_run"}
    try:
        config = load_runtime_config(args.runtime_config)
        configure_runtime(config)
        report.update(
            runtime_config_sha256=sha256(args.runtime_config),
            host=__import__("platform").node(),
            affinity=sorted(os.sched_getaffinity(0)),
            source_identity=source_identity(),
        )
        stage = "construction"
        root, _, manifest = model_root(args.model, args.backend == "native", config)
        report["fixture"] = manifest
        stage = "forward"
        random.seed(config["seed"])
        np.random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        model = root.build(root)
        initial_state = {name: tensor_record(value) for name, value in model.state_dict().items()}
        batch = torch.linspace(-1, 1, steps=math.prod(config["input_params"]["shape"])).reshape(
            config["input_params"]["shape"]
        )
        with torch.no_grad():
            output = model(batch)
        if not torch.isfinite(output).all():
            raise ValueError("Model forward returned nonfinite values")
        report["forward"] = {
            "status": "passed",
            "training": model.training,
            "autograd": False,
            "initial_state": initial_state,
            "final_state": {
                name: tensor_record(value) for name, value in model.state_dict().items()
            },
            "output": tensor_record(output),
            "finite_output": True,
            "parameter_bytes": sum(
                value.numel() * value.element_size() for value in model.parameters()
            ),
            "buffer_bytes": sum(value.numel() * value.element_size() for value in model.buffers()),
            "module_count": sum(1 for _ in model.modules()),
            "rng_after": rng_record(),
            "comparison_rule": "Exact tensor byte hashes under matched initialization and one-thread CPU execution; zero tolerance.",
        }
        report["status"] = "passed"
    except Exception as error:
        report.update(
            status=stage + "_error",
            stage=stage,
            error={
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            },
        )
    save(args.output, report)


def audit_model(args):
    config = load_runtime_config(args.runtime_config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=args.model + "-", dir=args.output.parent))
    rows = []
    for name in ("reference", "native"):
        destination = work / (name + ".json")
        command = [
            sys.executable,
            "-m",
            "tests.reference.table42",
            "--_audit-worker",
            "--backend",
            name,
            "--runtime-config",
            str(args.runtime_config.resolve()),
            "--model",
            args.model,
            "--output",
            str(destination),
        ]
        env = dict(
            os.environ,
            OMP_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            NUMEXPR_NUM_THREADS="1",
            PYTHONHASHSEED=str(config["seed"]),
        )
        with (work / (name + ".log")).open("w") as log:
            try:
                process = subprocess.run(
                    command, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=120, check=False
                )
                row = (
                    json.loads(destination.read_text())
                    if destination.exists()
                    else {"status": "worker_error", "returncode": process.returncode}
                )
            except subprocess.TimeoutExpired:
                row = {"status": "timeout", "stage": "validation_worker"}
        row.update(backend=name, command=command, log=str(work / (name + ".log")))
        rows.append(row)
    passed = all(row["status"] == "passed" for row in rows)
    fixture_equal = passed and rows[0]["fixture"]["runtime"] == rows[1]["fixture"]["runtime"]
    forward_equal = passed and rows[0]["forward"] == rows[1]["forward"]
    report = {
        "schema": "rcswx-table42-model-audit-v1",
        "model": args.model,
        "status": "passed" if fixture_equal and forward_equal else "failed",
        "runtime_config_sha256": sha256(args.runtime_config),
        "fixture_equal": bool(fixture_equal),
        "forward_equal": bool(forward_equal),
        "rows": rows,
    }
    save(args.output, report)
    print(
        json.dumps(
            {key: report[key] for key in ("model", "status", "fixture_equal", "forward_equal")}
        )
    )
    return 0 if report["status"] == "passed" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolve-config", type=Path)
    parser.add_argument("--admit-supplied-factories", action="store_true")
    parser.add_argument("--cpu", type=int, default=24)
    parser.add_argument("--runtime-config", type=Path)
    parser.add_argument("--model", choices=tuple(assets()["models"]))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--_audit-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--backend", choices=("reference", "native"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.resolve_config:
        save(args.resolve_config, resolve_runtime_config(args.cpu, args.admit_supplied_factories))
        print(f"Resolved diagnostic configuration: {args.resolve_config}")
        return 0
    if not args.runtime_config or not args.model or not args.output:
        parser.error("--runtime-config, --model and --output are required for a model audit")
    if args._audit_worker:
        audit_worker(args)
        return 0
    return audit_model(args)


if __name__ == "__main__":
    raise SystemExit(main())
