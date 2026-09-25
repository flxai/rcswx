"""Paired CPU network memory measurements in fresh, bounded subprocesses.

RSS runs have no allocation profiler, Python trace/profile hook or target warmup.
Linux VmHWM is reset after genotype preparation. Heap runs are separate Memray
native-trace processes: allocator-backed memory, not Rust-only memory or RSS.
"""

import argparse
import contextlib
import fnmatch
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

PHASES = ("align", "raw", "validated", "build")


def dump(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def number(value):
    """Represent nonfinite scalar observations explicitly in strict JSON."""
    if value is None:
        return None
    value = float(value)
    if math.isfinite(value):
        return value
    return {"nonfinite": "nan" if math.isnan(value) else "+inf" if value > 0 else "-inf"}


def resident():
    fields = {}
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith(("VmRSS:", "VmHWM:")):
            name, number, _ = line.split()
            fields[name.rstrip(":")] = int(number) * 1024
    return fields


def rng_record():
    import numpy as np
    import torch

    state = np.random.get_state()
    return digest(
        (random.getstate(), state[0], state[1].tolist(), *state[2:], torch.get_rng_state().tolist())
    )


def model_record(model, output, batch):
    """Hash values, gradients and buffers only after the measurement window."""
    import torch

    checksum = hashlib.sha256()

    def tensor(name, value):
        checksum.update(name.encode())
        if value is None:
            checksum.update(b"None")
        else:
            array = value.detach().contiguous().numpy()
            checksum.update(str((array.dtype, array.shape)).encode())
            checksum.update(memoryview(array))

    tensor("output", output)
    tensor("input_gradient", batch.grad)
    for name, value in model.state_dict().items():
        tensor("state/" + name, value)
    for name, parameter in model.named_parameters():
        tensor("gradient/" + name, parameter.grad)
    return {
        "digest": checksum.hexdigest(),
        "output_shape": list(output.shape),
        "finite_output": bool(torch.isfinite(output).all()),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "parameter_bytes": sum(
            parameter.numel() * parameter.element_size() for parameter in model.parameters()
        ),
    }


def outcome_record(phase, result, native, records_dir=None):
    from tests.reference.original import operation_record, tree_record

    if phase == "align":
        paths = result.paths if native else result.matrix[-1][-1].paths
        histories = hashlib.sha256()
        stream = (
            (records_dir / "histories.jsonl").open("w")
            if records_dir is not None
            else contextlib.nullcontext()
        )
        with stream as destination:
            for path in paths:
                encoded = json.dumps([operation_record(op) for op in path]) + "\n"
                histories.update(encoded.encode())
                if destination is not None:
                    destination.write(encoded)
        operations = [operation_record(op) for op in result.operations]
        if records_dir is not None:
            dump(records_dir / "operations.json", operations)
        return {
            "distance": number(result.distance),
            "path_count": len(paths),
            "histories_digest": histories.hexdigest(),
            "operations_digest": digest(operations),
            "matrix_shape": [len(result.model_ops1), len(result.model_ops2)],
        }
    if phase == "raw":
        root, selected, operations, *distances = result
        return {
            "child": tree_record(root),
            "selected": [operation_record(op) for op in selected],
            "operations": [operation_record(op) for op in operations],
            "distances": distances,
        }
    if phase == "validated":
        root, report = result
        return {
            "child": tree_record(root),
            "report": {
                key: [operation_record(op) for op in value]
                if key in {"crossover_operations", "crossover_all_operations"}
                else value
                for key, value in report.items()
            },
        }
    return [model_record(*item) for item in result]


def request(case, parents, builder, guard, native, phase, collapse_corners=False, workers=1):
    import torch

    from tests.reference.original import load

    reference = load()
    if phase == "align":
        if native:
            from rcswx import Alignment
        else:
            Alignment = reference.algorithm.AlignmentMatrixRecursive
        options = {"workers": workers} if native and workers != 1 else {}
        return lambda: Alignment(
            *parents, collapse_corners=collapse_corners, limiter=guard, **options
        )
    if phase == "raw":
        if native:
            from rcswx import raw_crossover
        else:
            raw_crossover = reference.algorithm.recursive_constrained_smith_waterman_crossover
        options = {"sampler": "reference"} if native else {}
        return lambda: raw_crossover(*parents, limiter=guard, **options)
    if phase == "validated":
        if native:
            from rcswx import validated_crossover

            return lambda: validated_crossover(
                *parents,
                rebuild=builder.re_id,
                batch_shape=guard.batch_shape,
                limiter=guard,
                sampler="reference",
            )
        return lambda: builder.recursive_constrained_smith_waterman_crossover(*parents)

    def build():
        results = []
        for root in parents:
            model = root.build(root)
            batch = (
                torch.linspace(-1, 1, steps=math.prod(case.shape))
                .reshape(case.shape)
                .requires_grad_()
            )
            output = model(batch)
            output.square().mean().backward()
            results.append((model, output, batch))
        return results

    return build


def worker(args):
    work = args.work_dir
    table42 = args.suite == "table42"
    collapse = args.collapse_corners == "on"
    progress = {
        "stage": "preparing",
        "request": {
            "case": args.case[0],
            "backend": args.backend,
            "phase": args.phase[0],
            "collapse_corners": collapse,
            "repeat": args._repeat,
            "workers": args.workers if args.backend == "native" else 1,
        },
    }
    dump(work / "progress.json", progress)
    try:
        if table42:
            raw_config = json.loads(args.runtime_config.read_text())
            os.sched_setaffinity(0, {raw_config["affinity_cpu"]})
        import numpy as np
        import torch

        from tests.reference.networks import network_pairs, prepare_pair
        from tests.reference.original import tree_record

        torch.set_num_threads(1)
        # Profiler initialization precedes seeding, as in the existing heap campaign.
        tracker = contextlib.nullcontext()
        if args.measure == "heap":
            import memray

            tracker = memray.Tracker(
                destination=memray.FileDestination(
                    work / "allocations.bin", compress_on_exit=False
                ),
                native_traces=True,
                trace_python_allocators=False,
                file_format=memray.FileFormat.AGGREGATED_ALLOCATIONS,
            )
        native = args.backend == "native"
        config, fixture_manifest = None, None
        if table42:
            from tests.reference import table42 as fixtures

            config = fixtures.load_runtime_config(args.runtime_config)
            fixtures.configure_runtime(config)
            case = fixtures.table_pairs(config)[args.case[0]]
            tree_record = fixtures.runtime_tree_record
            progress["request"].update(
                stratum=config["stratum"],
                profile_id=config["profile_id"],
                construction_seed=config["construction_seed"],
                call_seed=config["call_seed"],
                fixed_corner_threshold=0.25,
                runtime_config_sha256=fixtures.sha256(args.runtime_config),
                source_identity=fixtures.source_identity(),
            )
            dump(work / "progress.json", progress)
            parents, builder, guard, fixture_manifest = fixtures.prepare_pair(case, native, config)
        else:
            random.seed(71)
            np.random.seed(71)
            torch.manual_seed(71)
            case = network_pairs(args.suite)[args.case[0]]
            parents, builder, guard = prepare_pair(case, native)
        if args.reverse:
            parents = tuple(reversed(parents))
        invoke = request(
            case, parents, builder, guard, native, args.phase[0], collapse, args.workers
        )
        if native and args.affinity_cpus:
            os.sched_setaffinity(0, set(args.affinity_cpus))
        nodes = [len(list(root.serialise())) for root in parents]
        if sys.getprofile() is not None or sys.gettrace() is not None:
            raise RuntimeError("Worker must not inherit Python trace/profile hooks")
    except Exception as error:
        failure = {
            "status": "construction_error" if table42 else "worker_error",
            "stage": "preparing",
            "error": {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            },
            "request": progress["request"],
        }
        dump(work / "result.json", failure)
        return
    parents_before = [tree_record(root) for root in parents]
    if table42:
        random.seed(config["call_seed"])
        np.random.seed(config["call_seed"])
        torch.manual_seed(config["call_seed"])
        guard.timer.start()
    rng_before = rng_record()
    identity = {
        **progress["request"],
        "fixture_manifest": fixture_manifest,
        "affinity": sorted(os.sched_getaffinity(0)),
        "host": platform.node(),
    }
    progress.update(stage="measuring", nodes=nodes, identity=identity)
    dump(work / "progress.json", progress)
    gc.collect()
    reset_error = None
    if args.measure == "rss":
        try:
            Path("/proc/self/clear_refs").write_text("5\n")
        except OSError as error:
            reset_error = str(error)
    baseline = resident()
    result, error_record = None, None
    with tracker:
        started = time.perf_counter()
        try:
            result = invoke()
        except Exception as error:
            error_record = {"type": type(error).__name__, "message": str(error)}
        seconds = time.perf_counter() - started
        retained = resident() if args.measure == "rss" else None
    measurement = {
        "seconds": seconds,
        "align_wall_seconds": seconds if args.phase[0] == "align" else None,
        "internal_compute_time_seconds": number(getattr(result, "compute_time", None)),
        "profiled": args.measure == "heap",
    }
    execution_json = getattr(getattr(result, "_native", None), "execution_json", None)
    if execution_json is not None:
        measurement["native_execution"] = json.loads(execution_json())
    if args.measure == "rss":
        measurement.update(
            baseline_rss_bytes=baseline["VmRSS"],
            retained_rss_bytes=retained["VmRSS"],
            rss_peak_reset_error=reset_error,
            rss_peak_bytes=retained["VmHWM"] if reset_error is None else None,
            rss_peak_delta_bytes=max(0, retained["VmHWM"] - baseline["VmRSS"])
            if reset_error is None
            else None,
            lifetime_peak_rss_bytes=retained["VmHWM"],
            retained_rss_delta_bytes=retained["VmRSS"] - baseline["VmRSS"],
            after_release_rss_bytes=None,
        )
    scalar = {
        "status": "raised" if error_record else "returned",
        "distance": number(result.distance)
        if result is not None and args.phase[0] == "align"
        else None,
        "error": error_record,
    }
    progress.update(
        stage="fingerprinting",
        measurement=measurement,
        scalar_outcome=scalar,
        semantic_verification="unverified",
    )
    dump(work / "progress.json", progress)
    try:
        payload = (
            None
            if error_record
            else outcome_record(args.phase[0], result, native, work if table42 else None)
        )
        parents_after = [tree_record(root) for root in parents]
        rng_after = rng_record()
        outcome = {
            "status": scalar["status"],
            "error": error_record,
            "digest": digest(
                (
                    payload,
                    parents_before,
                    parents_after,
                    rng_before,
                    rng_after,
                    error_record["type"] if error_record else None,
                )
            ),
            "parents_before_digest": digest(parents_before),
            "parents_after_digest": digest(parents_after),
            "rng_before": rng_before,
            "rng_after": rng_after,
        }
        if args.phase[0] in {"align", "build"} and payload is not None:
            outcome["details"] = payload
        if table42:
            dump(work / "parent-effects.json", {"before": parents_before, "after": parents_after})
        result = None
        gc.collect()
        if args.measure == "rss":
            measurement["after_release_rss_bytes"] = resident()["VmRSS"]
        else:
            progress.update(stage="reading_capture", outcome=outcome)
            dump(work / "progress.json", progress)
            with memray.FileReader(work / "allocations.bin") as reader:
                measurement.update(
                    heap_peak_bytes=reader.metadata.peak_memory,
                    heap_retained_bytes=sum(
                        record.size for record in reader.get_leaked_allocation_records()
                    ),
                    memray_version=memray.__version__,
                    capture=str(work / "allocations.bin"),
                )
        dump(
            work / "result.json",
            {
                "status": "complete",
                "stage": "complete",
                "nodes": nodes,
                "measurement": measurement,
                "outcome": outcome,
                "identity": identity,
                "semantic_verification": "full",
                "semantic_scope": "Ordered returned histories, edits/dependencies, parent state/IDs and Python/NumPy/Torch RNG",
            },
        )
    except Exception as error:
        progress.update(
            status="fingerprint_error",
            error={
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            },
        )
        dump(work / "result.json", progress)


def run_worker(args, case, phase, measure, backend, repeat, captures, on_spawn=None):
    import psutil

    mode = getattr(args, "collapse_corners", "off")
    work = Path(
        tempfile.mkdtemp(prefix=f"{case.name}-{phase}-{measure}-{backend}-{mode}-", dir=captures)
    )
    command = [
        sys.executable,
        "-m",
        "benchmarks.memory",
        "--_worker",
        "--suite",
        args.suite,
        "--case",
        case.name,
        "--phase",
        phase,
        "--measure",
        measure,
        "--backend",
        backend,
        "--work-dir",
        str(work),
        "--collapse-corners",
        mode,
        "--_repeat",
        str(repeat),
    ]
    if args.reverse:
        command.append("--reverse")
    if getattr(args, "runtime_config", None) is not None:
        command.extend(("--runtime-config", str(args.runtime_config.resolve())))
    if backend == "native":
        command.extend(("--workers", str(getattr(args, "workers", 1))))
        if getattr(args, "affinity_cpus", None):
            command.append("--affinity-cpus")
            command.extend(map(str, args.affinity_cpus))
    child_environment = dict(
        os.environ,
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
    )
    if getattr(args, "runtime_config", None) is not None:
        child_environment["PYTHONHASHSEED"] = str(
            json.loads(args.runtime_config.read_text())["seed"]
        )
    started, sampled_peak, status = time.monotonic(), 0, None
    with (work / "worker.log").open("w") as log:
        process = subprocess.Popen(
            command, env=child_environment, stdout=log, stderr=subprocess.STDOUT
        )
        try:
            if on_spawn is not None:
                on_spawn(process.pid, work)
            monitor = psutil.Process(process.pid)
            while process.poll() is None:
                try:
                    sampled_peak = max(sampled_peak, monitor.memory_info().rss)
                except psutil.NoSuchProcess:
                    break
                if sampled_peak > args.rss_limit_mib * 1024**2:
                    status = "rss_limit"
                elif time.monotonic() - started > args.timeout:
                    status = "timeout"
                if status is not None:
                    break
                time.sleep(0.01)
        finally:
            if process.poll() is None:
                process.kill()
            returncode = process.wait()
    row = {
        "case": case.name,
        "phase": phase,
        "measure": measure,
        "backend": backend,
        "repeat": repeat,
        "reverse": args.reverse,
        "collapse_corners": mode == "on",
        "runtime_config_sha256": hashlib.sha256(args.runtime_config.read_bytes()).hexdigest()
        if getattr(args, "runtime_config", None) is not None
        else None,
        "stratum": "paired_current_reference" if args.suite == "table42" else None,
        "command": command,
        "worker_limits": {"timeout_seconds": args.timeout, "rss_limit_mib": args.rss_limit_mib},
        "returncode": returncode,
        "worker_seconds": time.monotonic() - started,
        "sampled_lifetime_peak_rss_bytes": sampled_peak,
        "work_dir": str(work),
    }
    if status is None and returncode == 0 and (work / "result.json").exists():
        row.update(json.loads((work / "result.json").read_text()))
    else:
        row["status"] = status or "worker_error"
        row["partial"] = (
            json.loads((work / "progress.json").read_text())
            if (work / "progress.json").exists()
            else {"stage": "startup"}
        )
        with (work / "worker.log").open("rb") as log:
            log.seek(max(0, os.fstat(log.fileno()).st_size - 4000))
            row["diagnostic"] = log.read().decode(errors="replace")
        if "measurement" in row["partial"]:
            row["measurement"] = row["partial"]["measurement"]
            row["scalar_outcome"] = row["partial"].get("scalar_outcome")
        row["interrupted_stage"] = row["partial"].get("stage", "startup")
        row["semantic_verification"] = "unverified"
    row["worker_wall_seconds"] = row["worker_seconds"]
    return row


def comparisons(rows):
    groups = {}
    for row in rows:
        key = (
            row["case"],
            row["phase"],
            row["measure"],
            row["repeat"],
            row["reverse"],
            row.get("collapse_corners", False),
            row.get("runtime_config_sha256"),
            row.get("stratum"),
        )
        group = groups.setdefault(key, {})
        if row["backend"] in group:
            raise ValueError(f"Duplicate backend/repetition in benchmark observations: {key}")
        group[row["backend"]] = row
    result = []
    for key, pair in groups.items():
        case, phase, measure, repeat, reverse, collapse, config_hash, stratum = key
        complete = set(pair) == {"reference", "native"} and all(
            row["status"] == "complete"
            and isinstance(row.get("outcome", {}).get("digest"), str)
            and row.get("semantic_verification", "full") == "full"
            for row in pair.values()
        )
        equal = (
            pair["reference"]["outcome"]["digest"] == pair["native"]["outcome"]["digest"]
            if complete
            else None
        )
        identities_match = complete and (
            pair["reference"].get("identity", {}).get("host")
            == pair["native"].get("identity", {}).get("host")
            and pair["reference"].get("identity", {}).get("affinity")
            == pair["native"].get("identity", {}).get("affinity")
            and pair["reference"].get("identity", {}).get("source_identity")
            == pair["native"].get("identity", {}).get("source_identity")
            and pair["reference"].get("worker_limits") == pair["native"].get("worker_limits")
        )
        eligible = bool(
            equal
            and identities_match
            and all(
                row["outcome"]["status"] == "returned"
                and not row.get("measurement", {}).get("profiled", False)
                and (
                    phase != "align"
                    or (
                        type(row["outcome"].get("details", {}).get("distance")) in (int, float)
                        and math.isfinite(row["outcome"]["details"]["distance"])
                    )
                )
                for row in pair.values()
            )
        )
        result.append(
            {
                "case": case,
                "phase": phase,
                "measure": measure,
                "repeat": repeat,
                "reverse": reverse,
                "collapse_corners": collapse,
                "runtime_config_sha256": config_hash,
                "stratum": stratum,
                "semantics_equal": equal,
                "identities_equal": bool(identities_match),
                "speedup_eligible": eligible,
                "status": ("match" if equal else "mismatch") if complete else "incomplete",
            }
        )
    return result


def controller(args):
    if args.suite == "table42":
        from benchmarks.table42_campaign import controller as table42_controller

        return table42_controller(args)
    import rcswx
    import torch
    from rcswx import _core

    from tests.reference.networks import network_pairs
    from tests.reference.original import environment

    cases = {
        name: case
        for name, case in network_pairs(args.suite).items()
        if not args.case or any(fnmatch.fnmatchcase(name, pattern) for pattern in args.case)
    }
    if not cases:
        raise SystemExit("No cases match the requested patterns")
    if args.list:
        for case in cases.values():
            print(
                f"{case.name:32} {case.family:14} {case.category:20} scale={case.scale} shape={case.shape}"
            )
        return 0
    if args.output is None:
        raise SystemExit("--output is required unless --list is used")
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    captures = (args.captures_dir or args.output.with_suffix(".captures")).resolve()
    captures.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    package = Path(rcswx.__file__).parent
    report = {
        "schema": 1,
        "environment": environment(),
        "host": platform.node(),
        "native_sha256": hashlib.sha256(Path(_core.__file__).read_bytes()).hexdigest(),
        "package_sha256": {
            str(path.relative_to(package)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(package.rglob("*.py"))
        },
        "harness_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                Path(__file__),
                Path("tests/reference/networks.py"),
                Path("tests/reference/corpus.py"),
            )
        },
        "protocol": {
            "seed": 71,
            "threads": 1,
            "fresh_process_per_call": True,
            "warmup_calls": 0,
            "rss": "Unprofiled. Linux VmHWM reset after imports/genotype preparation; fingerprinting excluded.",
            "heap": "Separate Memray native-trace run; allocator-backed new live bytes, not Rust-only memory. Existing allocator pools and GPU memory are outside this measurement.",
            "build": "Both parent models: construction, forward, squared-mean backward; outputs/models/gradients retained at the boundary.",
            "after_release": "Diagnostic after fingerprinting and releasing the returned object; input genotypes remain live. Not a leak detector.",
            "limits": "Timeout/RSS limits cover the entire worker, including startup/profiling/fingerprinting. Partial stage is recorded; missing pairs are not speedups.",
            "rss_limit_mib": args.rss_limit_mib,
            "timeout_seconds": args.timeout,
            "profiler": importlib.metadata.version("memray") if args.measure != "rss" else None,
        },
        "cases": [case.record() for case in cases.values()],
        "rows": [],
        "comparisons": [],
    }
    dump(args.output, report)
    measures = ("rss", "heap") if args.measure == "both" else (args.measure,)
    for case in cases.values():
        for phase in args.phase or PHASES:
            for measure in measures:
                for mode in (
                    ("off", "on") if args.collapse_corners == "both" else (args.collapse_corners,)
                ):
                    worker_args = argparse.Namespace(**vars(args))
                    worker_args.collapse_corners = mode
                    for repeat in range(args.repeats):
                        # Alternate order to avoid giving one backend every cold first slot.
                        backends = (
                            ("reference", "native") if repeat % 2 == 0 else ("native", "reference")
                        )
                        for backend in backends:
                            row = run_worker(
                                worker_args, case, phase, measure, backend, repeat, captures
                            )
                            report["rows"].append(row)
                            report["comparisons"] = comparisons(report["rows"])
                            dump(args.output, report)
                            print(
                                f"{case.name} {phase} {measure} {backend} collapse={mode}: {row['status']} {row.get('outcome', {}).get('status', '')}",
                                flush=True,
                            )
    return int(
        any(row["status"] == "worker_error" for row in report["rows"])
        or any(pair["status"] == "mismatch" for pair in report["comparisons"])
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("smoke", "scaling", "all", "table42"), default="smoke")
    parser.add_argument("--case", action="append", help="Repeatable case-name glob")
    parser.add_argument("--phase", action="append", choices=PHASES)
    parser.add_argument("--measure", choices=("rss", "heap", "both"), default="both")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--rss-limit-mib", type=float, default=1024)
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--captures-dir", type=Path)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--collapse-corners", choices=("off", "on", "both"), default="off")
    parser.add_argument("--runtime-config", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--affinity-cpus", type=int, nargs="+")
    parser.add_argument("--wall-budget-seconds", type=float)
    parser.add_argument("--budget-file", type=Path)
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--backend", choices=("reference", "native"), help=argparse.SUPPRESS)
    parser.add_argument("--work-dir", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--_repeat", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if (
        args.repeats < 1
        or not math.isfinite(args.timeout)
        or args.timeout <= 0
        or not math.isfinite(args.rss_limit_mib)
        or args.rss_limit_mib <= 0
    ):
        parser.error("repeats, timeout and RSS limit must be finite and positive")
    if args.workers < 1:
        parser.error("workers must be positive")
    if args.workers != 1 or args.affinity_cpus is not None:
        if args.suite != "table42" or args.phase not in (None, ["align"]):
            parser.error("Explicit workers and affinity require Table 4.2 alignment")
        if args.affinity_cpus is None or len(set(args.affinity_cpus)) < args.workers:
            parser.error("Provide at least one distinct affinity CPU per requested worker")
        if len(set(args.affinity_cpus)) != len(args.affinity_cpus) or min(args.affinity_cpus) < 0:
            parser.error("Affinity CPUs must be distinct nonnegative indices")
    if args._worker and args.collapse_corners == "both":
        parser.error("A worker must receive one explicit corner-collapse mode")
    if args.suite == "table42":
        if args.runtime_config is None:
            parser.error("table42 requires --runtime-config with resolved provenance")
        args.phase = args.phase or ["align"]
        if args.phase != ["align"] or args.measure != "rss" or args.reverse:
            parser.error(
                "Primary table42 supports forward align/RSS only; diagnostics are separate"
            )
        if (
            not args._worker
            and not args.list
            and (
                args.budget_file is None
                or args.wall_budget_seconds is None
                or not math.isfinite(args.wall_budget_seconds)
                or args.wall_budget_seconds <= 0
            )
        ):
            parser.error(
                "table42 requires a persistent --budget-file and positive --wall-budget-seconds"
            )
        if args.timeout > 1800 or args.repeats > 3:
            parser.error(
                "Table 4.2 workers are bounded to 1800 seconds and three requested successes"
            )
    elif (
        args.runtime_config is not None
        or args.budget_file is not None
        or args.wall_budget_seconds is not None
    ):
        parser.error("Resolved configuration and campaign budgets apply only to table42")
    elif args.collapse_corners != "off" and (args.phase or list(PHASES)) != ["align"]:
        parser.error("Corner collapse is an alignment-only experiment axis")
    if args._worker:
        worker(args)
    else:
        raise SystemExit(controller(args))


if __name__ == "__main__":
    main()
