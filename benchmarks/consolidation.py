"""Frozen-corpus consolidation evidence, reusing the archived bounded worker loop.

Run each package generation in its own interpreter/environment. Pair selection,
explicit masks, identical-request repetition, and sampler seeds are independent
inputs. Primary timings never enable stage hooks; full fingerprints and cleanup
are outside their measurement window. Prepared private pickles remain external.
"""

import argparse
import contextlib
import functools
import gc
import hashlib
import importlib.util
import io
import json
import math
import multiprocessing
import os
import platform
import random
import statistics
import sys
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path

from .consolidation_pool import ASSIGNMENT, assigned_cpu, run_pool
from .memory import digest, dump, number, outcome_record, resident, rng_record

SCHEMA = "rcswx-consolidation-v2"
PHASES = (
    "distance",
    "align",
    "control",
    "apply",
    "enumerate",
    "probabilities",
    "draw",
    "raw",
    "validated",
)
STOCHASTIC = {"draw", "raw", "validated"}
NUMERICAL = {"probabilities", *STOCHASTIC}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def archive_paths(archive):
    archive = Path(archive).resolve()
    snapshot = archive / "snapshot"
    if not snapshot.is_dir():
        snapshot = archive
    return snapshot, snapshot / "pairs" / "manifest.json", archive / "results.json"


def load_archive(archive):
    snapshot, manifest_path, results_path = archive_paths(archive)
    manifest = json.loads(manifest_path.read_text())
    historical = json.loads(results_path.read_text())
    metadata = historical["metadata"]
    if not metadata.get("complete"):
        raise ValueError("The historical corpus run is incomplete")
    if metadata["input_sha256"] != manifest["input_sha256"]:
        raise ValueError("Historical results and prepared pairs refer to different inputs")
    pairs = manifest["pairs"]
    if len({pair["id"] for pair in pairs}) != len(pairs):
        raise ValueError("Duplicate prepared pair IDs")
    for pair in pairs:
        # The archived manifest contains absolute paths from its original host.
        # Relocate by basename only, then verify bytes against its recorded hash.
        path = snapshot / "pairs" / Path(pair["path"]).name
        if sha256(path) != pair["sha256"]:
            raise ValueError(f"Prepared parent snapshot changed: {pair['id']}")
        pair["path"] = str(path)
    return (
        snapshot,
        manifest,
        historical,
        {
            "input_sha256": manifest["input_sha256"],
            "manifest_sha256": sha256(manifest_path),
            "historical_results_sha256": sha256(results_path),
            "normalization_sha256": metadata["normalization_sha256"],
        },
    )


def dataset_runner(snapshot, historical):
    runner_path = snapshot / "benchmark_dataset.py"
    if sha256(runner_path) != historical["metadata"]["runner_sha256"]:
        raise ValueError("Archived bounded-worker harness hash changed")
    if sha256(snapshot / "prepare_dataset.py") != historical["metadata"]["normalization_sha256"]:
        raise ValueError("Archived normalization/resolver hash changed")
    os.environ["RCSWX_REPO_ROOT"] = str(Path(__file__).resolve().parents[1])
    sys.path.insert(0, str(snapshot))
    spec = importlib.util.spec_from_file_location("rcswx_archived_dataset", runner_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def package_snapshot():
    import rcswx
    from rcswx import _core

    package = Path(rcswx.__file__).parent
    files = {
        str(path.relative_to(package)): sha256(path)
        for path in sorted(package.rglob("*"))
        if path.is_file() and path.suffix in {".py", ".pyi", ".so", ".pyd"}
    }
    return {
        "path": str(package),
        "files": files,
        "digest": digest(files),
        "extension_path": _core.__file__,
        "extension_sha256": sha256(_core.__file__),
        "consolidated_engine": hasattr(_core, "NativePrepared"),
    }


def runtime_identity():
    from tests.reference.original import environment

    return {
        "host": platform.node(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "runtime": environment(),
        "affinity": sorted(os.sched_getaffinity(0)),
        "thread_environment": {
            key: os.environ.get(key)
            for key in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
    }


def initialize_runtime(cpu):
    import numpy  # noqa: F401 -- warm imports belong outside timed workers
    import rcswx
    import scipy.stats  # noqa: F401
    import torch
    from rcswx import genotype, pipeline, sampling  # noqa: F401

    from tests.reference.original import load

    os.sched_setaffinity(0, {cpu})
    torch.set_num_threads(1)
    load()
    # Resolve lazy public imports before fork, without warming any target call.
    for name in (
        "Alignment",
        "distance",
        "raw_crossover",
        "validated_crossover",
        "Reconstructor",
        "PCFG",
    ):
        getattr(rcswx, name)


def quantiles(items, count, key):
    ordered = sorted(items, key=key)
    if not ordered:
        return []
    if count == 1:
        return [ordered[len(ordered) // 2]]
    return [ordered[round(index * (len(ordered) - 1) / (count - 1))] for index in range(count)]


def structure_features(parents):
    counts = Counter()

    def visit(node):
        children = tuple(visit(child) for child in node.children)
        if len(node.children) > 2:
            counts["branched_nodes"] += 1
        if len(children) > 1 and len(set(children)) < len(children):
            counts["repeated_sibling_signatures"] += 1
        return node.operation.name, children

    for parent in parents:
        visit(parent)
    return dict(counts)


def select_pairs(manifest, historical, features, count=40):
    """Deterministic engineering strata, not a population-frequency sample."""
    if not 30 <= count <= 50:
        raise ValueError("Diagnostic selection must contain 30–50 recorded pairs")
    pairs = {pair["id"]: pair for pair in manifest["pairs"]}
    selected = {}

    def add(items, reason):
        for item in items:
            identifier = item["id"] if "id" in item else item["pair"]
            if identifier not in selected and len(selected) == count:
                continue
            selected.setdefault(identifier, []).append(reason)

    add(
        quantiles(list(pairs.values()), count // 4, lambda item: (item["nodes"], item["id"])),
        "size_quantile",
    )
    native_rows = [row for row in historical["rows"] if row["backend"] == "native"]
    for phase in ("distance", "raw"):
        completed = [row for row in native_rows if row["phase"] == phase and row["status"] == "ok"]
        add(
            quantiles(completed, count // 8, lambda row: (row["cpu_seconds"], row["pair"])),
            phase + "_latency_quantile",
        )
        speedups = [
            row
            for row in historical["comparisons"]
            if row["phase"] == phase and row.get("speedup") is not None
        ]
        add(
            quantiles(speedups, count // 10, lambda row: (row["speedup"], row["pair"])),
            phase + "_speedup_quantile",
        )
    limited = [row for row in native_rows if row["status"] not in {"ok", "exception"}]
    add(
        quantiles(limited, count // 12, lambda row: (row["nodes"], row["pair"], row["phase"])),
        "resource_limited",
    )
    repeated = [
        pairs[name] for name, value in features.items() if value.get("repeated_sibling_signatures")
    ]
    add(
        quantiles(repeated, count // 12, lambda item: (item["nodes"], item["id"])),
        "tied_history_candidate_repeated_siblings",
    )
    branched = [pairs[name] for name, value in features.items() if value.get("branched_nodes")]
    add(
        quantiles(branched, count // 12, lambda item: (item["nodes"], item["id"])),
        "branched_structure",
    )
    remaining = [pair for identifier, pair in pairs.items() if identifier not in selected]
    add(
        quantiles(remaining, count - len(selected), lambda item: (item["nodes"], item["id"])),
        "size_gap_fill",
    )
    return [
        {
            "id": name,
            "sha256": pairs[name]["sha256"],
            "nodes": pairs[name]["nodes"],
            "reasons": reasons,
            "structure": features.get(name, {}),
        }
        for name, reasons in selected.items()
    ]


def reference_probabilities(values, skewness):
    """The frozen sampling.py law, split only for isolated numerical measurement."""
    import numpy as np
    from scipy.stats import skewnorm

    distribution = skewnorm(skewness)
    sample_at = np.linspace(distribution.ppf(0.01), distribution.ppf(0.99), int(values[-1] * 4))
    samples = distribution.pdf(sample_at)
    probabilities = [samples[int(value * 4) - 1] for value in values]
    probabilities /= np.sum(probabilities)
    return probabilities


def exception_record(error):
    frames = traceback.extract_tb(error.__traceback__)
    return {
        "type": type(error).__name__,
        "message": str(error),
        "location": {
            "file": frames[-1].filename,
            "line": frames[-1].lineno,
            "function": frames[-1].name,
        }
        if frames
        else None,
    }


def prepare_target(phase, parents, options, auxiliary):
    import numpy as np
    import rcswx
    from rcswx import sampling

    from tests.reference.original import operation_record

    guard = parents[0].limiter
    modern = hasattr(rcswx._core, "NativePrepared")
    plan_options = {"profile": options["profile"]} if modern else {}
    sampler_options = {"sampler": options["sampler"]} if modern else {}
    if options["sampler"] == "native" and phase in STOCHASTIC:
        stream = rcswx._core.NativeRng(options["sampler_seed"])
        auxiliary["native_rng"] = stream
        sampler_options["rng"] = stream
    if phase == "distance":
        return lambda: rcswx.distance(*parents)
    if phase == "align":
        return lambda: rcswx.Alignment(*parents, limiter=guard, **plan_options)
    if phase == "raw":
        return lambda: rcswx.raw_crossover(
            *parents, skewness=options["skewness"], limiter=guard, **sampler_options
        )
    if phase == "validated":
        from rcswx.grammars import einspace

        shape = tuple(parents[0].input_params["shape"])
        builder = rcswx.Reconstructor(rcswx.PCFG(einspace.grammar, guard), guard, "iterative")
        auxiliary["rebuild_attempts"] = 0

        def rebuild(root):
            auxiliary["rebuild_attempts"] += 1
            return builder.re_id(root)

        return lambda: rcswx.validated_crossover(
            *parents,
            rebuild=rebuild,
            batch_shape=shape,
            skewness=options["skewness"],
            limiter=guard,
            **sampler_options,
        )
    if phase == "control":

        def control():
            plan = rcswx.Alignment(*parents, limiter=guard, **plan_options)
            auxiliary["plan"] = plan
            operations = plan.nontrivial_ops
            masks, costs = sampling.valid_combinations(operations)
            return {
                "mask": masks[len(masks) // 2],
                "operation_count": len(operations),
                "operations_digest": digest([operation_record(op) for op in operations]),
                "masks_digest": digest(masks),
                "costs_digest": digest(costs),
            }

        return control
    plan = rcswx.Alignment(*parents, limiter=guard, **plan_options)
    auxiliary["plan"] = plan
    operations = plan.nontrivial_ops
    auxiliary["operations_digest"] = digest([operation_record(op) for op in operations])
    if phase == "apply":
        mask = options["control"]["mask"]
        if len(mask) != len(operations) or set(mask) - {"0", "1"}:
            raise ValueError("Frozen mask does not fit this plan")
        if auxiliary["operations_digest"] != options["control"]["operations_digest"]:
            raise ValueError("Operation sequence differs from the frozen control plan")
        selected = [
            operation for bit, operation in zip(mask, operations, strict=True) if bit == "1"
        ]
        auxiliary["selected"] = selected
        return lambda: plan.generate_offspring(selected)
    if phase == "enumerate":
        return lambda: sampling.valid_combinations(operations)
    masks, costs = sampling.valid_combinations(operations)
    auxiliary.update(masks=masks, costs=costs)
    if phase == "probabilities":
        if options["sampler"] == "native":
            return lambda: sampling.probabilities(costs, options["skewness"])
        return lambda: reference_probabilities(costs, options["skewness"])
    # Draw comparisons share exactly the frozen reference weights. They do not
    # require equal outcomes from NumPy and ChaCha12 streams.
    weights = reference_probabilities(costs, options["skewness"])
    weight_values = weights.tolist()
    auxiliary["weights_digest"] = digest(weight_values)
    if options["sampler"] == "native":
        return lambda: masks[rcswx._core.native_draw(weight_values, auxiliary["native_rng"])]
    return lambda: str(np.random.choice(masks, p=weights))


def phase_record(phase, result, auxiliary, options):
    from tests.reference.original import operation_record, tree_record

    if phase == "distance":
        return number(result)
    if phase in {"align", "raw", "validated"}:
        return outcome_record(phase, result, True)
    if phase == "control":
        return result
    if phase == "apply":
        return {
            "child": tree_record(result),
            "mask": options["control"]["mask"],
            "operations_digest": auxiliary["operations_digest"],
            "selected": [operation_record(op) for op in auxiliary["selected"]],
            "matches_frozen_plan": auxiliary["operations_digest"]
            == options["control"]["operations_digest"],
        }
    if phase == "enumerate":
        masks, costs = result
        return {
            "masks_digest": digest(masks),
            "costs_digest": digest(costs),
            "mask_count": len(masks),
            "operations_digest": auxiliary["operations_digest"],
        }
    common = {
        "masks_digest": digest(auxiliary["masks"]),
        "costs_digest": digest(auxiliary["costs"]),
        "mask_count": len(auxiliary["masks"]),
        "operations_digest": auxiliary["operations_digest"],
    }
    if phase == "probabilities":
        return {**common, "probabilities": [number(value) for value in result]}
    return {**common, "selected_mask": result, "weights_digest": auxiliary["weights_digest"]}


def measured_worker(connection, pair, generation, phase, cpu, *, runner, options):
    try:
        sys.stdout = open(os.devnull, "w")
        import numpy as np
        import torch

        from benchmarks.reference import profile_stages
        from tests.reference.original import tree_record

        os.sched_setaffinity(0, {cpu})
        connection.send({"stage": "preparing", "worker_affinity": sorted(os.sched_getaffinity(0))})
        raw = Path(pair["path"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != pair["sha256"]:
            raise ValueError("Prepared parent snapshot changed")
        parents = runner.ParentUnpickler(io.BytesIO(raw), True).load()
        guard = parents[0].limiter
        guard.__init__(dict(guard.limits))
        random.seed(options["sampler_seed"])
        np.random.seed(options["sampler_seed"])
        torch.manual_seed(options["sampler_seed"])
        flattened = [parent.serialise() for parent in parents]
        counts = list(map(len, flattened))
        if counts != [pair["nodes"], pair["nodes"]]:
            raise ValueError("Prepared node counts changed")
        noop = flattened[0] == flattened[1] and all(
            a.operation.name == b.operation.name for a, b in zip(*flattened, strict=True)
        )
        auxiliary, details = {}, {}
        try:
            invoke = prepare_target(phase, parents, options, auxiliary)
        except Exception as error:
            outcome = {
                "value": None,
                "parents": [tree_record(parent) for parent in parents],
                "rng": rng_record(),
                "exception_type": type(error).__name__,
            }
            if "native_rng" in auxiliary:
                outcome["native_rng"] = auxiliary["native_rng"].state()
            connection.send(
                {
                    "stage": "complete",
                    "status": "setup_exception",
                    "error": exception_record(error),
                    "outcome_digest": digest(outcome),
                    "outcome": outcome,
                }
            )
            return
        if sys.gettrace() is not None or sys.getprofile() is not None:
            raise RuntimeError("Workers must start without trace/profile hooks")
        gc.collect()
        error = result = None
        profiling = (
            profile_stages(True, details=details)
            if options["profile"]
            else contextlib.nullcontext({})
        )
        with profiling as stages:
            Path("/proc/self/clear_refs").write_text("5\n")
            baseline = resident()["VmRSS"]
            connection.send(
                {
                    "stage": "measuring",
                    "setup_cpu_seconds": time.process_time(),
                    "baseline_rss_bytes": baseline,
                    "noop": noop,
                }
            )
            wall, process_cpu, thread_cpu = (
                time.perf_counter(),
                time.process_time(),
                time.thread_time(),
            )
            try:
                result = invoke()
            except Exception as failure:
                error = exception_record(failure)
            thread_cpu = time.thread_time() - thread_cpu
            process_cpu = time.process_time() - process_cpu
            wall = time.perf_counter() - wall
            memory = resident()
        measurement = {
            "cpu_seconds": thread_cpu,
            "process_cpu_seconds": process_cpu,
            "wall_seconds": wall,
            "rss_peak_bytes": memory["VmHWM"],
            "rss_peak_delta_bytes": max(0, memory["VmHWM"] - baseline),
            "profiled": options["profile"],
        }
        connection.send({"stage": "fingerprinting", **measurement, "error": error})
        payload = None if error else phase_record(phase, result, auxiliary, options)
        outcome = {
            "value": payload,
            "parents": [tree_record(parent) for parent in parents],
            "rng": rng_record(),
            "exception_type": error["type"] if error else None,
        }
        if "native_rng" in auxiliary:
            outcome["native_rng"] = auxiliary["native_rng"].state()
        plan = result if phase == "align" and not error else auxiliary.get("plan")
        plans = details.get("plans", [])
        if plan is not None and all(plan is not item for item in plans):
            plans.append(plan)
        workloads = []
        for item in plans:
            workloads.append(
                {
                    "token_counts": [len(item.model_ops1), len(item.model_ops2)],
                    "native_timings": getattr(item, "timings", {}) if options["profile"] else {},
                    "retained_histories": len(item.paths),
                    "returned_steps": sum(map(len, item.paths)),
                    "edits": len(item.nontrivial_ops),
                    "native_counters": getattr(item, "stats", {}),
                }
            )
        if workloads:
            details["workloads"] = workloads
        details.pop("plans", None)
        if phase == "validated":
            details["rebuild_attempts"] = auxiliary["rebuild_attempts"]
        status = "exception" if error else "ok"
        if phase == "apply" and not error and not payload["matches_frozen_plan"]:
            status = "control_plan_mismatch"
        # Do not include cleanup in the public target-call metric. Diagnostic
        # retention and cyclic Python metadata can change its cost materially.
        connection.send({"stage": "cleanup", "status": status, "outcome_digest": digest(outcome)})
        cleanup_start = time.perf_counter()
        plans.clear()
        plan = item = None
        del result, invoke, parents, flattened
        auxiliary.clear()
        collected = gc.collect()
        cleanup = {
            "wall_seconds": time.perf_counter() - cleanup_start,
            "gc_collected": collected,
            "rss_bytes": resident()["VmRSS"],
        }
        connection.send(
            {
                "stage": "complete",
                "status": status,
                "outcome_digest": digest(outcome),
                "outcome": outcome,
                "details": details,
                "stages": {
                    name: {"inclusive_seconds": value[0], "calls": value[1]}
                    for name, value in stages.items()
                },
                "cleanup": cleanup,
            }
        )
    except BaseException as failure:
        connection.send(
            {
                "stage": "failed",
                "status": "harness_error",
                "error": {**exception_record(failure), "traceback": traceback.format_exc()},
            }
        )
    finally:
        connection.close()


def select_command(args):
    snapshot, manifest, historical, identity = load_archive(args.archive)
    initialize_runtime(args.cpu)
    runner = dataset_runner(snapshot, historical)
    features = {}
    for pair in manifest["pairs"]:
        with open(pair["path"], "rb") as source:
            parents = runner.ParentUnpickler(source, True).load()
        features[pair["id"]] = structure_features(parents)
    result = {
        "schema": SCHEMA,
        "kind": "diagnostic-selection",
        "corpus": identity,
        "policy": "Deterministic size, latency, speedup, resource and branching strata. Repeated siblings are tie candidates, not a claim of measured retained histories.",
        "pairs": select_pairs(manifest, historical, features, args.count),
    }
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite a frozen selection manifest")
    dump(args.output, result)
    print(
        json.dumps(
            {
                "selection": str(args.output),
                "sha256": sha256(args.output),
                "pairs": len(result["pairs"]),
            }
        )
    )


def controls_from(path, identity):
    if path is None:
        return {}, None
    source = json.loads(path.read_text())
    if source["metadata"]["generation"] != "baseline" or source["metadata"]["corpus"] != identity:
        raise ValueError("Explicit controls must be frozen baseline results for this corpus")
    if not all(
        source["metadata"].get(key)
        for key in ("complete", "package_unchanged", "harness_unchanged")
    ):
        raise ValueError("Explicit controls require a complete, unchanged baseline campaign")
    controls = {}
    for row in source["rows"]:
        if row["phase"] != "control" or row["status"] != "ok":
            continue
        value = row["outcome"]["value"]
        if row["pair"] in controls and controls[row["pair"]] != value:
            raise ValueError("Baseline control changed across repetitions")
        controls[row["pair"]] = value
    return controls, sha256(path)


def requests_for(pairs, args):
    index = 0
    for pair in pairs:
        for seed in args.sampler_seeds if args.sampler_seeds is not None else [pair["seed"]]:
            for repeat in range(args.repeats):
                for phase in args.phases:
                    yield {
                        "request_index": index,
                        "pair": pair,
                        "phase": phase,
                        "repeat_id": repeat,
                        "sampler_seed": seed,
                    }
                    index += 1


def execute_request(request, cpu, *, runner, context, args, controls):
    pair, phase = request["pair"], request["phase"]
    options = {
        "sampler": args.sampler,
        "sampler_seed": request["sampler_seed"],
        "skewness": args.skewness,
        "profile": args.profile,
        "control": controls.get(pair["id"]),
    }
    # Each lane owns its forked copy of args and the archived module.
    args.cpu = cpu
    runner.worker = functools.partial(measured_worker, runner=runner, options=options)
    if phase == "apply" and options["control"] is None:
        row = {
            "pair": pair["id"],
            "phase": phase,
            "status": "unavailable_control",
            "stage": "complete",
        }
    else:
        row = runner.run_worker(context, pair, args.generation, phase, args)
    if row.get("worker_affinity", [cpu]) != [cpu]:
        raise RuntimeError("Measured worker escaped its assigned CPU")
    row.pop("backend", None)
    row.update(
        generation=args.generation,
        pair_sha256=pair["sha256"],
        pair_seed=pair["seed"],
        repeat_id=request["repeat_id"],
        sampler=args.sampler if phase in NUMERICAL else "none",
        sampler_seed=request["sampler_seed"],
        skewness=args.skewness,
        profiled=args.profile,
        control_digest=digest(options["control"])
        if phase == "apply" and options["control"] is not None
        else None,
    )
    return row


def aggregate_memory_limit():
    """Record the effective cgroup-v2 ceiling, including inherited limits."""
    for line in Path("/proc/self/cgroup").read_text().splitlines():
        if line.startswith("0::"):
            root = Path("/sys/fs/cgroup")
            group = root / line[3:].lstrip("/")
            limits = []
            while group.is_relative_to(root):
                path = group / "memory.max"
                if path.exists():
                    value = path.read_text().strip()
                    if value != "max":
                        limits.append(int(value))
                if group == root:
                    break
                group = group.parent
            return min(limits) if limits else None
    return None


def run_command(args):
    cpus = getattr(args, "cpus", None)
    if cpus is not None:
        available = os.sched_getaffinity(0)
        if (
            not cpus
            or len(set(cpus)) != len(cpus)
            or args.cpu in cpus
            or not set([args.cpu, *cpus]).issubset(available)
        ):
            raise ValueError(
                "Parallel workers need distinct available CPUs and a separate controller CPU"
            )
    snapshot, manifest, historical, corpus = load_archive(args.archive)
    initialize_runtime(args.cpu)
    runner = dataset_runner(snapshot, historical)
    package = package_snapshot()
    if args.generation == "baseline":
        recorded = {
            name.rsplit("/rcswx/", 1)[1]: value
            for name, value in historical["metadata"]["native_code_sha256"].items()
        }
        if any(package["files"].get(name) != value for name, value in recorded.items()):
            raise ValueError(
                "Baseline package does not match the recorded historical source/binary hashes"
            )
        if package["extension_sha256"] not in {
            value for name, value in recorded.items() if name.endswith((".so", ".pyd"))
        }:
            raise ValueError("The active extension is not the measured historical binary")
        if args.sampler != "reference":
            raise ValueError("The historical baseline has only the reference sampler")
    elif not package["consolidated_engine"]:
        raise ValueError("Consolidated generation requires the installed native structural engine")
    build_record = json.loads(args.build_record.read_text()) if args.build_record else None
    if (
        build_record is not None
        and build_record.get("extension_sha256") != package["extension_sha256"]
    ):
        raise ValueError("Build provenance belongs to another native extension")
    if args.generation == "consolidated" and build_record is None:
        raise ValueError(
            "Consolidated evidence requires --build-record with exact compiler/source/artifact provenance"
        )
    if args.sampler == "native" and any(phase not in NUMERICAL for phase in args.phases):
        raise ValueError(
            "Native-sampler campaigns measure numerical/stochastic phases, not duplicate deterministic rows"
        )
    if "apply" in args.phases and args.controls is None:
        raise ValueError("Explicit application requires a frozen baseline control result file")
    pairs = manifest["pairs"]
    selection_hash = None
    if args.selection:
        selection = json.loads(args.selection.read_text())
        if selection["corpus"] != corpus:
            raise ValueError("Selection manifest belongs to another frozen corpus")
        names = {pair["id"]: pair["sha256"] for pair in selection["pairs"]}
        if len(names) != len(selection["pairs"]):
            raise ValueError("Duplicate selected pair IDs")
        pairs = [pair for pair in pairs if pair["id"] in names]
        if len(pairs) != len(names) or any(pair["sha256"] != names[pair["id"]] for pair in pairs):
            raise ValueError("Selected pair IDs/hashes do not match the corpus")
        selection_hash = sha256(args.selection)
    controls, controls_hash = controls_from(args.controls, corpus)
    identity = runtime_identity()
    expected_runtime = historical["metadata"]["reference"]
    for key in ("python", "numpy", "scipy", "torch", "source_hashes"):
        if identity["runtime"][key] != expected_runtime[key]:
            raise ValueError(f"Pinned benchmark runtime/reference mismatch: {key}")
    if identity["host"] != historical["metadata"]["host"]:
        raise ValueError("Full corpus evidence must run on its recorded host")
    if (
        (cpus is None and args.cpu != historical["metadata"]["affinity_cpu"])
        or args.timeout != historical["metadata"]["timeout_wall_seconds"]
        or args.memory_gib * 1024**3 != historical["metadata"]["memory_limit_bytes"]
    ):
        raise ValueError("CPU affinity and external limits must match the historical cohort")
    harness_paths = [
        Path(__file__),
        Path(__file__).with_name("consolidation_pool.py"),
        Path(__file__).with_name("reference.py"),
        Path(__file__).with_name("memory.py"),
        Path(__file__).parents[1] / "tests/reference/original.py",
    ]
    harness = {
        str(path.relative_to(Path(__file__).parents[1])): sha256(path) for path in harness_paths
    }
    seeds_per_pair = len(args.sampler_seeds) if args.sampler_seeds is not None else 1
    metadata = {
        "schema": SCHEMA,
        "complete": False,
        "generation": args.generation,
        "baseline_kind": "verified-historical-binary" if args.generation == "baseline" else None,
        "corpus": corpus,
        "selection_sha256": selection_hash,
        "controls_sha256": controls_hash,
        "identity": identity,
        "package": package,
        "harness": harness,
        "executable": sys.executable,
        "build_provenance": build_record,
        "historical_build_provenance": (
            "Package source and measured binary hashes are verified; the historical archive does not record compiler flags."
            if args.generation == "baseline" and build_record is None
            else None
        ),
        "phases": args.phases,
        "sampler": args.sampler,
        "skewness": args.skewness,
        "repeat_count": args.repeats,
        "sampler_seeds": args.sampler_seeds,
        "seed_policy": "explicit sampler seeds"
        if args.sampler_seeds is not None
        else "recorded pair seed, unchanged across repeat_id",
        "profiled": args.profile,
        "worker_limits": {
            "measuring_wall_seconds": args.timeout,
            "other_stage_wall_seconds": 90,
            "process_memory_bytes": int(args.memory_gib * 1024**3),
        },
        "scope": "Fresh forked worker, no target warmup. Input adaptation and materialization are included in distance/align/raw/validated calls. Application/enumeration/numerics/draw prepare plans outside the timed call. Application includes materialization, but not control validation. Fingerprinting and cleanup are separate. Diagnostic spans overlap and must not be summed; native plan setup timings are outside these narrow phase metrics.",
        "rss_scope": "Reset VmHWM immediately before call; total and incremental RSS exclude fingerprinting. Polling peaks are lower bounds for censored calls, not completed VmHWM.",
        "control_policy": "The middle enumeration-order legal mask is frozen with the baseline operation sequence before application.",
        "planned_requests": len(pairs) * len(args.phases) * args.repeats * seeds_per_pair,
        "pairs": [{"id": pair["id"], "sha256": pair["sha256"]} for pair in pairs],
    }
    if cpus is not None:
        metadata["execution"] = {
            "mode": "parallel",
            "workers": len(cpus),
            "worker_cpus": cpus,
            "controller_cpu": args.cpu,
            "assignment": ASSIGNMENT,
            "aggregate_memory_limit_bytes": aggregate_memory_limit(),
        }
    args.output.mkdir(parents=True, exist_ok=False)
    dump(args.output / "metadata.json", metadata)
    context = multiprocessing.get_context("fork")
    execute = functools.partial(
        execute_request, runner=runner, context=context, args=args, controls=controls
    )
    requests = requests_for(pairs, args)
    started = time.monotonic()
    if cpus is not None:
        run_pool(list(requests), execute, cpus, args.output)
        row_count = metadata["planned_requests"]
    else:
        row_count = 0
        with (args.output / "rows.jsonl").open("x") as stream:
            for request in requests:
                row = execute(request, args.cpu)
                row.update(request_index=request["request_index"], worker_cpu=args.cpu)
                stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                stream.flush()
                del row
                row_count += 1
                # Do not retain outcome payloads in a controller that will fork.
                if row_count % 100 == 0:
                    print(
                        json.dumps(
                            {
                                "completed": row_count,
                                "planned": metadata["planned_requests"],
                                "seconds": time.monotonic() - started,
                            }
                        ),
                        flush=True,
                    )
    metadata.update(
        complete=row_count == metadata["planned_requests"],
        elapsed_seconds=time.monotonic() - started,
        package_unchanged=package_snapshot() == package,
        harness_unchanged=all(
            sha256(Path(__file__).parents[1] / name) == value for name, value in harness.items()
        ),
    )
    with (args.output / "rows.jsonl").open() as stream:
        rows = [json.loads(line) for line in stream]
    statuses = Counter(row["status"] for row in rows)
    report = {"metadata": metadata, "rows": rows, "status_counts": dict(statuses)}
    dump(args.output / "results.json", report)
    dump(args.output / "metadata.json", metadata)
    if not metadata["package_unchanged"] or not metadata["harness_unchanged"]:
        raise RuntimeError("Inputs changed during the campaign; recorded measurements are stale")
    print(
        json.dumps(
            {
                "results": str(args.output / "results.json"),
                "rows": len(rows),
                "status_counts": report["status_counts"],
            }
        )
    )


def cohort_identity(metadata):
    """Package generation and sampler differ deliberately; experimental inputs do not."""
    identity = {
        key: metadata[key]
        for key in (
            "corpus",
            "identity",
            "selection_sha256",
            "worker_limits",
            "harness",
            "profiled",
            "skewness",
            "seed_policy",
            "sampler_seeds",
        )
    }
    identity["execution"] = metadata.get("execution")
    return identity


def probability_agreement(first, second):
    common = ("masks_digest", "costs_digest", "mask_count", "operations_digest")
    if any(first.get(key) != second.get(key) for key in common):
        return {"match": False, "reason": "different distribution inputs"}
    left, right = first.get("probabilities", []), second.get("probabilities", [])
    if (
        not left
        or len(left) != len(right)
        or any(
            type(value) not in (int, float) or not math.isfinite(value) for value in (*left, *right)
        )
    ):
        return {
            "match": False,
            "reason": "missing, nonfinite or unequal-length probability vectors",
        }
    differences = [abs(a - b) for a, b in zip(left, right, strict=True)]
    total_variation = sum(differences) / 2
    normalized = all(value >= 0 for value in (*left, *right)) and all(
        abs(sum(vector) - 1) <= 1e-12 for vector in (left, right)
    )
    match = (
        normalized
        and total_variation < 1e-9
        and all(
            difference <= 1e-12 + 1e-10 * abs(reference)
            for reference, difference in zip(left, differences, strict=True)
        )
    )
    return {
        "match": match,
        "max_absolute_error": max(differences),
        "total_variation": total_variation,
        "normalized": normalized,
    }


def nontrivial(row):
    value = row.get("outcome", {}).get("value")
    phase = row["phase"]
    if phase == "distance":
        return type(value) in (int, float) and math.isfinite(value) and value > 0
    if phase == "align":
        distance = value.get("distance") if isinstance(value, dict) else None
        return type(distance) in (int, float) and math.isfinite(distance) and distance > 0
    if phase in {"raw", "validated"}:
        return not row.get("noop", False)
    if phase == "apply":
        return bool(value and "1" in value["mask"])
    if phase in {"enumerate", "probabilities", "draw"}:
        return bool(value and value["mask_count"] > 1)
    return False


def compare_pair(first, second, comparison, phase):
    pair = {
        "comparison": comparison,
        "phase": phase,
        "statuses": [row["status"] if row is not None else "missing" for row in (first, second)],
        "semantic_match": None,
        "speedup_eligible": False,
    }
    if first is None or second is None:
        return pair
    complete = all(
        row["status"] in {"ok", "exception"}
        and isinstance(row.get("outcome_digest"), str)
        and row.get("stage") == "complete"
        for row in (first, second)
    )
    if not complete:
        return pair
    if comparison == "baseline-reference_vs_consolidated-reference":
        pair["semantic_match"] = first["outcome_digest"] == second["outcome_digest"]
    elif phase == "probabilities":
        if first["status"] == second["status"] == "exception":
            pair["semantic_match"] = first["outcome_digest"] == second["outcome_digest"]
        elif first["status"] == second["status"] == "ok":
            pair["probability_agreement"] = probability_agreement(
                first["outcome"]["value"], second["outcome"]["value"]
            )
            pair["semantic_match"] = pair["probability_agreement"]["match"]
        else:
            pair["semantic_match"] = False
    elif phase == "draw":
        if first["status"] == second["status"] == "ok":
            left, right = first["outcome"]["value"], second["outcome"]["value"]
            pair["distribution_inputs_match"] = all(
                left.get(key) == right.get(key)
                for key in ("masks_digest", "costs_digest", "operations_digest", "weights_digest")
            )
            pair["semantic_match"] = pair["distribution_inputs_match"]
            pair["outcome_contract"] = (
                "Same weighted categorical work; different RNG outcomes are expected"
            )
    returned = all(row["status"] == "ok" for row in (first, second))
    pair["speedup_eligible"] = bool(
        pair["semantic_match"]
        and returned
        and first.get("worker_cpu") == second.get("worker_cpu")
        and all(
            row["_campaign_complete"] and not row["profiled"] and nontrivial(row)
            for row in (first, second)
        )
        and all(row.get("cpu_seconds", 0) > 0 for row in (first, second))
    )
    if pair["speedup_eligible"]:
        pair["cpu_speedup"] = first["cpu_seconds"] / second["cpu_seconds"]
        if second.get("rss_peak_bytes", 0) > 0:
            pair["total_rss_ratio"] = first["rss_peak_bytes"] / second["rss_peak_bytes"]
        if second.get("rss_peak_delta_bytes", 0) > 0:
            pair["incremental_rss_ratio"] = (
                first["rss_peak_delta_bytes"] / second["rss_peak_delta_bytes"]
            )
    return pair


def compare_reports(reports):
    groups, cohorts, natural, generations = {}, {}, defaultdict(list), {}
    for report in reports:
        metadata = report["metadata"]
        if metadata.get("schema") != SCHEMA:
            raise ValueError("Not a consolidation result file")
        if not metadata.get("package_unchanged") or not metadata.get("harness_unchanged"):
            raise ValueError("Refusing stale or unverified campaign inputs")
        execution = metadata.get("execution")
        if execution is not None:
            cpus = execution.get("worker_cpus", [])
            if (
                execution.get("mode") != "parallel"
                or execution.get("assignment") != ASSIGNMENT
                or not cpus
                or len(set(cpus)) != len(cpus)
                or execution.get("workers") != len(cpus)
                or execution.get("controller_cpu") in cpus
            ):
                raise ValueError("Invalid parallel execution provenance")
        cohort = digest(cohort_identity(metadata))
        cohorts[cohort] = cohort_identity(metadata)
        generation_key = cohort, metadata["generation"]
        package_digest = metadata["package"]["digest"]
        if generation_key in generations and generations[generation_key] != package_digest:
            raise ValueError("Different package artifacts are mixed within a generation/cohort")
        generations[generation_key] = package_digest
        for source_row in report["rows"]:
            row = dict(source_row, _campaign_complete=metadata["complete"])
            if (
                row["generation"] != metadata["generation"]
                or row["profiled"] != metadata["profiled"]
            ):
                raise ValueError("Row labels disagree with their provenance")
            if execution is not None:
                cpu = assigned_cpu(
                    row["pair"], row["phase"], row["repeat_id"], row["sampler_seed"], cpus
                )
                if row.get("worker_cpu") != cpu or row.get("worker_affinity", [cpu]) != [cpu]:
                    raise ValueError("Row CPU disagrees with deterministic lane assignment")
            key = (
                cohort,
                row["pair"],
                row["pair_sha256"],
                row["phase"],
                row["repeat_id"],
                row["sampler_seed"],
                row["control_digest"],
            )
            slot = row["generation"], row["sampler"]
            group = groups.setdefault(key, {})
            if slot in group:
                raise ValueError(f"Duplicate generation/sampler observation: {key}, {slot}")
            group[slot] = row
            if row["phase"] in {"raw", "validated"}:
                natural[(cohort, row["phase"], *slot, bool(row.get("noop", False)))].append(row)
    comparisons = []
    for key, group in groups.items():
        cohort, pair_id, pair_hash, phase, repeat, sampler_seed, control_digest = key
        sampler = "reference" if phase in NUMERICAL else "none"
        candidates = [
            (
                "baseline-reference_vs_consolidated-reference",
                group.get(("baseline", sampler)),
                group.get(("consolidated", sampler)),
            )
        ]
        if phase in {"probabilities", "draw"} and ("consolidated", "native") in group:
            candidates.append(
                (
                    "baseline-reference_vs_consolidated-native",
                    group.get(("baseline", "reference")),
                    group[("consolidated", "native")],
                )
            )
            if ("consolidated", "reference") in group:
                candidates.append(
                    (
                        "consolidated-reference_vs_consolidated-native",
                        group[("consolidated", "reference")],
                        group[("consolidated", "native")],
                    )
                )
        for name, first, second in candidates:
            comparisons.append(
                {
                    "cohort": cohort,
                    "pair": pair_id,
                    "pair_sha256": pair_hash,
                    "repeat_id": repeat,
                    "sampler_seed": sampler_seed,
                    "control_digest": control_digest,
                    **compare_pair(first, second, name, phase),
                }
            )
    ratios = defaultdict(list)
    for comparison in comparisons:
        if comparison["speedup_eligible"]:
            ratios[(comparison["cohort"], comparison["phase"], comparison["comparison"])].append(
                comparison
            )
    summaries = [
        {
            "cohort": key[0],
            "phase": key[1],
            "comparison": key[2],
            "eligible_requests": len(rows),
            "unique_pairs": len({row["pair"] for row in rows}),
            "median_cpu_speedup": statistics.median(row["cpu_speedup"] for row in rows),
            "median_total_rss_ratio": statistics.median(
                row["total_rss_ratio"] for row in rows if "total_rss_ratio" in row
            )
            if any("total_rss_ratio" in row for row in rows)
            else None,
        }
        for key, rows in ratios.items()
    ]
    natural_summaries = []
    for key, rows in natural.items():
        successful = [row for row in rows if row["status"] == "ok" and not row["profiled"]]
        child_digests = Counter(digest(row["outcome"]["value"]["child"]) for row in successful)
        natural_summaries.append(
            {
                "cohort": key[0],
                "phase": key[1],
                "generation": key[2],
                "sampler": key[3],
                "noop": key[4],
                "requests": len(rows),
                "statuses": dict(Counter(row["status"] for row in rows)),
                "exceptions": dict(
                    Counter(
                        row.get("error", {}).get("type")
                        for row in rows
                        if row["status"] in {"exception", "setup_exception"}
                    )
                ),
                "median_returned_cpu_seconds": statistics.median(
                    row["cpu_seconds"] for row in successful
                )
                if successful
                else None,
                "median_returned_wall_seconds": statistics.median(
                    row["wall_seconds"] for row in successful
                )
                if successful
                else None,
                "distinct_child_outcomes": len(child_digests),
                "child_outcome_counts": dict(child_digests),
                "rebuild_attempts_completed": sum(
                    row.get("details", {}).get("rebuild_attempts", 0)
                    for row in rows
                    if row["status"] in {"ok", "exception"}
                ),
                "censored_retry_work": "unknown; never inferred from a timeout",
            }
        )
    return {
        "schema": SCHEMA,
        "cohorts": cohorts,
        "comparisons": comparisons,
        "summaries": summaries,
        "natural_outcomes": natural_summaries,
        "notes": [
            "Ratios use only complete, semantically compatible, nontrivial, unprofiled successful requests.",
            "Different native/reference RNG children are descriptive outcomes, never full-call paired speedups.",
            "Probability agreement uses absolute 1e-12, relative 1e-10 and total-variation <1e-9 tolerances.",
            "Repetitions share a pair and sampler state; pair-selection seeds are not repeat IDs.",
            "Different cohorts remain separate; no population weighting or products of median ratios.",
        ],
    }


def compare_command(args):
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite a comparison report")
    report = compare_reports([json.loads(path.read_text()) for path in args.results])
    report["inputs"] = [{"path": str(path), "sha256": sha256(path)} for path in args.results]
    dump(args.output, report)
    mismatches = sum(row["semantic_match"] is False for row in report["comparisons"])
    print(
        json.dumps(
            {"report": str(args.output), "mismatches": mismatches, "summaries": report["summaries"]}
        )
    )
    if mismatches:
        raise SystemExit(1)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    select = commands.add_parser("select")
    select.add_argument("--archive", type=Path, required=True)
    select.add_argument("--output", type=Path, required=True)
    select.add_argument("--cpu", type=int, default=16)
    select.add_argument("--count", type=int, default=40)
    compare = commands.add_parser("compare")
    compare.add_argument("results", type=Path, nargs="+")
    compare.add_argument("--output", type=Path, required=True)
    run = commands.add_parser("run")
    run.add_argument("--archive", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--generation", choices=("baseline", "consolidated"), required=True)
    run.add_argument("--selection", type=Path)
    run.add_argument("--controls", type=Path)
    run.add_argument("--build-record", type=Path)
    run.add_argument("--phases", nargs="+", choices=PHASES, default=["distance", "align", "raw"])
    run.add_argument("--sampler", choices=("reference", "native"), default="reference")
    run.add_argument("--sampler-seeds", type=int, nargs="+")
    run.add_argument("--skewness", type=float, default=0)
    run.add_argument("--repeats", type=int, default=1)
    run.add_argument("--profile", action="store_true")
    run.add_argument("--cpu", type=int, default=16)
    run.add_argument(
        "--cpus",
        type=int,
        nargs="+",
        help="Parallel worker CPUs; --cpu is the separate controller CPU. Creates a new cohort.",
    )
    run.add_argument("--timeout", type=float, default=30)
    run.add_argument("--memory-gib", type=float, default=2)
    args = parser.parse_args(argv)
    if args.command == "select":
        select_command(args)
        return
    if args.command == "compare":
        compare_command(args)
        return
    if args.repeats <= 0 or len(set(args.phases)) != len(args.phases):
        parser.error("Repetitions must be positive and phases must be unique")
    if args.sampler_seeds is not None and (
        len(set(args.sampler_seeds)) != len(args.sampler_seeds)
        or any(not 0 <= seed < 2**32 for seed in args.sampler_seeds)
    ):
        parser.error("Benchmark sampler seeds must be distinct integers in [0, 2**32)")
    if not math.isfinite(args.skewness):
        parser.error("Benchmark skewness must be finite")
    run_command(args)


if __name__ == "__main__":
    main()
