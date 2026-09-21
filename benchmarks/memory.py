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
from pathlib import Path

PHASES = ("align", "raw", "validated", "build")


def dump(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


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


def outcome_record(phase, result, native):
    from tests.reference.original import operation_record, tree_record

    if phase == "align":
        paths = result.paths if native else result.matrix[-1][-1].paths
        histories = hashlib.sha256()
        for path in paths:
            histories.update(json.dumps([operation_record(op) for op in path]).encode())
            histories.update(b"\n")
        return {
            "distance": float(result.distance),
            "path_count": len(paths),
            "histories_digest": histories.hexdigest(),
            "operations_digest": digest([operation_record(op) for op in result.operations]),
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


def request(case, parents, builder, guard, native, phase):
    import torch

    from tests.reference.original import load

    reference = load()
    if phase == "align":
        if native:
            from rcswx import Alignment
        else:
            Alignment = reference.algorithm.AlignmentMatrixRecursive
        return lambda: Alignment(*parents, limiter=guard)
    if phase == "raw":
        if native:
            from rcswx import raw_crossover
        else:
            raw_crossover = reference.algorithm.recursive_constrained_smith_waterman_crossover
        return lambda: raw_crossover(*parents, limiter=guard)
    if phase == "validated":
        if native:
            from rcswx import validated_crossover

            return lambda: validated_crossover(
                *parents, rebuild=builder.re_id, batch_shape=guard.batch_shape, limiter=guard
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
    import numpy as np
    import torch

    from tests.reference.networks import network_pairs, prepare_pair
    from tests.reference.original import tree_record

    work = args.work_dir
    progress = {"stage": "preparing"}
    dump(work / "progress.json", progress)
    tracker = contextlib.nullcontext()
    if args.measure == "heap":
        import memray

        tracker = memray.Tracker(
            destination=memray.FileDestination(work / "allocations.bin", compress_on_exit=False),
            native_traces=True,
            trace_python_allocators=False,
            file_format=memray.FileFormat.AGGREGATED_ALLOCATIONS,
        )
    torch.set_num_threads(1)
    random.seed(71)
    np.random.seed(71)
    torch.manual_seed(71)
    case = network_pairs(args.suite)[args.case[0]]
    native = args.backend == "native"
    parents, builder, guard = prepare_pair(case, native)
    if args.reverse:
        parents = tuple(reversed(parents))
    invoke = request(case, parents, builder, guard, native, args.phase[0])
    nodes = [len(root.serialise()) for root in parents]
    if sys.getprofile() is not None or sys.gettrace() is not None:
        raise RuntimeError("Worker must not inherit Python trace/profile hooks")
    progress.update(stage="measuring", nodes=nodes)
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
    measurement = {"seconds": seconds, "profiled": args.measure == "heap"}
    if args.measure == "rss":
        measurement.update(
            baseline_rss_bytes=baseline["VmRSS"],
            retained_rss_bytes=retained["VmRSS"],
            rss_peak_reset_error=reset_error,
            rss_peak_bytes=retained["VmHWM"] if reset_error is None else None,
            rss_peak_delta_bytes=max(0, retained["VmHWM"] - baseline["VmRSS"])
            if reset_error is None
            else None,
            retained_rss_delta_bytes=retained["VmRSS"] - baseline["VmRSS"],
        )
    progress.update(stage="fingerprinting", measurement=measurement)
    dump(work / "progress.json", progress)
    payload = None if error_record else outcome_record(args.phase[0], result, native)
    outcome = {
        "status": "raised" if error_record else "returned",
        "error": error_record,
        "digest": digest(
            (
                payload,
                [tree_record(root) for root in parents],
                rng_record(),
                error_record["type"] if error_record else None,
            )
        ),
    }
    if args.phase[0] in {"align", "build"} and payload is not None:
        outcome["details"] = payload
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
        {"status": "complete", "nodes": nodes, "measurement": measurement, "outcome": outcome},
    )


def run_worker(args, case, phase, measure, backend, repeat, captures):
    import psutil

    work = Path(tempfile.mkdtemp(prefix=f"{case.name}-{phase}-{measure}-{backend}-", dir=captures))
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
    ]
    if args.reverse:
        command.append("--reverse")
    started, sampled_peak, status = time.monotonic(), 0, None
    with (work / "worker.log").open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
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
    return row


def comparisons(rows):
    groups = {}
    for row in rows:
        key = row["case"], row["phase"], row["measure"], row["repeat"], row["reverse"]
        groups.setdefault(key, {})[row["backend"]] = row
    result = []
    for (case, phase, measure, repeat, reverse), pair in groups.items():
        if len(pair) != 2:
            continue
        completed = all(row["status"] == "complete" for row in pair.values())
        equal = (
            pair["reference"].get("outcome", {}).get("digest")
            == pair["native"].get("outcome", {}).get("digest")
            if completed
            else None
        )
        result.append(
            {
                "case": case,
                "phase": phase,
                "measure": measure,
                "repeat": repeat,
                "reverse": reverse,
                "semantics_equal": equal,
                "status": ("match" if equal else "mismatch") if completed else "incomplete",
            }
        )
    return result


def controller(args):
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
                for repeat in range(args.repeats):
                    # Alternate order to avoid giving one backend every cold first slot.
                    backends = (
                        ("reference", "native") if repeat % 2 == 0 else ("native", "reference")
                    )
                    for backend in backends:
                        row = run_worker(args, case, phase, measure, backend, repeat, captures)
                        report["rows"].append(row)
                        report["comparisons"] = comparisons(report["rows"])
                        dump(args.output, report)
                        print(
                            f"{case.name} {phase} {measure} {backend}: {row['status']} {row.get('outcome', {}).get('status', '')}",
                            flush=True,
                        )
    return int(
        any(row["status"] == "worker_error" for row in report["rows"])
        or any(pair["status"] == "mismatch" for pair in report["comparisons"])
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("smoke", "scaling", "all"), default="smoke")
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
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--backend", choices=("reference", "native"), help=argparse.SUPPRESS)
    parser.add_argument("--work-dir", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if (
        args.repeats < 1
        or not math.isfinite(args.timeout)
        or args.timeout <= 0
        or not math.isfinite(args.rss_limit_mib)
        or args.rss_limit_mib <= 0
    ):
        parser.error("repeats, timeout and RSS limit must be finite and positive")
    if args._worker:
        worker(args)
    else:
        raise SystemExit(controller(args))


if __name__ == "__main__":
    main()
