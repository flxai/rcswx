"""Measure a tagged distance implementation on the frozen population corpus.

Uses the archived input resolver and the existing consolidation provenance
helpers. Each public call runs in a fresh, bounded child; imports and parent
preparation are outside the measured interval. The v0.5 adapter supplies the
worker argument to the public distance wrapper's existing edit_path delegation.
"""

import argparse
import gc
import io
import json
import math
import multiprocessing
import os
import random
import sys
import time
import traceback
from pathlib import Path

from .consolidation import (
    dataset_runner,
    exception_record,
    initialize_runtime,
    load_archive,
    package_snapshot,
    runtime_identity,
    sha256,
)
from .memory import digest, dump, resident, rng_record


def measure(connection, started, pair, args, runner):
    try:
        sys.stdout = open(os.devnull, "w")
        import numpy as np
        import rcswx
        import torch

        from tests.reference.original import load, tree_record

        # Start every generation on the same performance core. Parallel calls
        # gain the complete allowed set immediately before measurement.
        os.sched_setaffinity(0, {args.cpus[0]})
        raw = Path(pair["path"]).read_bytes()
        if sha256(pair["path"]) != pair["sha256"]:
            raise ValueError("Prepared parent snapshot changed")
        native = args.version != "original"
        parents = runner.ParentUnpickler(io.BytesIO(raw), native).load()
        guard = parents[0].limiter
        guard.__init__(dict(guard.limits))
        random.seed(pair["seed"])
        np.random.seed(pair["seed"])
        torch.manual_seed(pair["seed"])
        flattened = [root.serialise() for root in parents]
        if list(map(len, flattened)) != [pair["nodes"], pair["nodes"]]:
            raise ValueError("Prepared parent node counts changed")
        noop = flattened[0] == flattened[1] and all(
            a.operation.name == b.operation.name for a, b in zip(*flattened, strict=True)
        )
        captured = []
        if args.version == "v0.5":
            import rcswx.api as api

            original_edit_path = api.edit_path

            def parallel_edit_path(*positional, **keywords):
                plan = original_edit_path(*positional, **keywords, workers=args.workers)
                captured.append(plan._native.execution_json())
                return plan

            api.edit_path = parallel_edit_path
        invoke = rcswx.distance if native else load().algorithm.rcswx_distance
        if sys.gettrace() is not None or sys.getprofile() is not None:
            raise RuntimeError("Timing workers must not have trace/profile hooks")
        gc.collect()
        os.sched_setaffinity(0, set(args.cpus))
        Path("/proc/self/clear_refs").write_text("5\n")
        baseline = resident()["VmRSS"]
        connection.send(
            {
                "stage": "measuring",
                "setup_cpu_seconds": time.process_time(),
                "baseline_rss_bytes": baseline,
                "worker_affinity": sorted(os.sched_getaffinity(0)),
                "noop": noop,
            }
        )
        error = result = None
        process_cpu, thread_cpu = time.process_time(), time.thread_time()
        wall = time.perf_counter()
        # Shared timestamp gives the supervisor a real wall-time observation
        # when the child cannot return (RSS/wall limit). No CPU/wall conversion.
        started.value = wall
        try:
            result = invoke(*parents)
        except Exception as failure:
            error = exception_record(failure)
        wall = time.perf_counter() - wall
        process_cpu = time.process_time() - process_cpu
        thread_cpu = time.thread_time() - thread_cpu
        memory = resident()
        connection.send(
            {
                "stage": "fingerprinting",
                "wall_seconds": wall,
                "process_cpu_seconds": process_cpu,
                "cpu_seconds": thread_cpu,
                "rss_peak_bytes": memory["VmHWM"],
                "rss_peak_delta_bytes": max(0, memory["VmHWM"] - baseline),
                "error": error,
            }
        )
        execution = json.loads(captured[0]) if captured else None
        if args.version == "v0.5" and not noop and error is None:
            if not execution or execution["worker_limit"] != args.workers:
                raise RuntimeError(f"Native worker ceiling differs from request: {execution}")
        value = None if error else float(result)
        outcome = {
            "value": value,
            "parents": [tree_record(root) for root in parents],
            "rng": rng_record(),
            "exception_type": error["type"] if error else None,
        }
        connection.send(
            {
                "stage": "complete",
                "status": "exception" if error else "ok",
                "distance": value,
                "outcome_digest": digest(outcome),
                "native_execution": execution,
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


def bounded_call(context, pair, args, runner):
    """Archived supervisor pattern, with an explicit wall-clock start record."""
    import psutil

    receiver, sender = context.Pipe(duplex=False)
    started = context.Value("d", 0.0)
    child = context.Process(target=measure, args=(sender, started, pair, args, runner))
    child.start()
    sender.close()
    watched = psutil.Process(child.pid)
    row = {
        "pair": pair["id"],
        "pair_sha256": pair["sha256"],
        "nodes": pair["nodes"],
        "seed": pair["seed"],
        "phase": "distance",
        "status": "starting",
        "stage": "preparing",
    }
    stage_started = time.monotonic()
    peak = 0
    measuring_peak = None
    try:
        while True:
            while receiver.poll():
                try:
                    message = receiver.recv()
                except EOFError:
                    break
                row.update(message)
                if message["stage"] == "measuring":
                    measuring_peak = message["baseline_rss_bytes"]
                stage_started = time.monotonic()
            if row["stage"] in {"complete", "failed"}:
                break
            if not child.is_alive():
                if receiver.poll():
                    continue
                row.update(status="worker_exit", exit_code=child.exitcode)
                break
            try:
                rss = watched.memory_info().rss
                peak = max(peak, rss)
                cpu = watched.cpu_times()
                elapsed_cpu = max(0, cpu.user + cpu.system - row.get("setup_cpu_seconds", 0))
            except psutil.NoSuchProcess:
                continue
            start = started.value
            measuring = row["stage"] == "measuring" and start > 0
            elapsed = time.perf_counter() - start if measuring else time.monotonic() - stage_started
            if measuring:
                measuring_peak = max(measuring_peak or 0, rss)
            if peak > args.memory_gib * 1024**3:
                status = "rss_limit" if measuring else "setup_or_fingerprint_rss_limit"
            elif elapsed > (args.timeout if measuring else 90):
                status = "timeout" if measuring else "setup_or_fingerprint_timeout"
            else:
                time.sleep(0.02)
                continue
            row.update(
                status=status,
                observed_wall_lower_bound=elapsed if measuring else None,
                observed_cpu_lower_bound=elapsed_cpu if measuring else None,
            )
            child.kill()
            break
    finally:
        child.join(timeout=5)
        if child.is_alive():
            child.kill()
            child.join()
        receiver.close()
    row["observed_peak_rss_bytes"] = peak
    row["observed_measuring_peak_rss_bytes"] = measuring_peak
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--version", choices=["original", "v0.1", "v0.3", "v0.5"], required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--cpus", type=int, nargs="+", required=True)
    parser.add_argument("--controller-cpu", type=int, default=31)
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--memory-gib", type=float, default=2)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    if args.version != "v0.5" and args.workers != 1:
        parser.error("Historical implementations are serial")
    if args.workers < 1 or len(set(args.cpus)) != len(args.cpus) or len(args.cpus) < args.workers:
        parser.error("Need distinct allowed CPUs for the requested worker count")
    if not set([*args.cpus, args.controller_cpu]) <= os.sched_getaffinity(0):
        parser.error("Requested CPUs are outside this process's affinity")
    if args.controller_cpu in args.cpus:
        parser.error("Controller CPU must be separate from measured CPUs")
    if not all(math.isfinite(value) and value > 0 for value in (args.timeout, args.memory_gib)):
        parser.error("Resource bounds must be positive and finite")
    if args.limit is not None and args.limit < 1:
        parser.error("Smoke limit must be positive")
    if args.output.exists():
        parser.error("Refusing to overwrite an existing campaign")
    artifact = json.loads(args.artifact.read_text())
    snapshot, manifest, historical, corpus = load_archive(args.archive)
    runner = dataset_runner(snapshot, historical)
    initialize_runtime(args.controller_cpu)
    import rcswx

    if args.version == "v0.5" and not getattr(rcswx._core, "PARALLEL_CAPABLE", False):
        parser.error("v0.5 requires the exact parallel-capable wheel")
    package = package_snapshot()
    if package["extension_sha256"] != artifact["extension_sha256"]:
        parser.error("Installed extension does not match the specified artifact")
    pairs = manifest["pairs"][: args.limit]
    harness = {
        str(path.relative_to(Path(__file__).parents[1])): sha256(path)
        for path in [
            Path(__file__),
            Path(__file__).with_name("consolidation.py"),
            Path(__file__).with_name("memory.py"),
        ]
    }
    identity = runtime_identity()
    metadata = {
        "schema": "rcswx-distance-campaign-v1",
        "complete": False,
        "host": identity["host"],
        "version": args.version,
        "workers": args.workers,
        "affinity_cpus": args.cpus,
        "controller_cpu": args.controller_cpu,
        "timeout_wall_seconds": args.timeout,
        "memory_limit_bytes": int(args.memory_gib * 1024**3),
        "pairs": len(pairs),
        "corpus_pairs": len(manifest["pairs"]),
        "population_histogram": historical["metadata"]["population_histogram"],
        "reference": historical["metadata"]["reference"],
        "runtime": identity,
        "corpus": corpus,
        "package": package,
        "artifact": artifact,
        "harness": harness,
        "metric": "wall_seconds",
        "call_boundary": "Public distance call, including shared-clock publication and v0.5 execution-report serialization. Imports, preparation and fingerprinting excluded; temporary-plan destruction follows the public API in every version.",
        "parallel_adapter": "v0.5 only: supply workers through the public distance wrapper's existing edit_path delegation; preserve its limiter and no-op shortcut.",
        "pool_startup": "Fresh forked measurement child per pair; native pool startup is included, never initialized in controller.",
        "numerical_threads": 1,
    }
    args.output.mkdir(parents=True)
    dump(args.output / "metadata.json", metadata)
    context = multiprocessing.get_context("fork")
    rows = []
    began = time.monotonic()
    with (args.output / "rows.jsonl").open("w") as stream:
        for pair in pairs:
            row = bounded_call(context, pair, args, runner)
            rows.append(row)
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            if len(rows) % 25 == 0 or len(rows) == len(pairs):
                print(
                    json.dumps(
                        {
                            "version": args.version,
                            "completed": len(rows),
                            "total": len(pairs),
                            "elapsed": time.monotonic() - began,
                        }
                    ),
                    flush=True,
                )
    metadata.update(
        complete=len(rows) == len(pairs),
        elapsed_seconds=time.monotonic() - began,
        package_unchanged=package_snapshot() == package,
        harness_unchanged=all(
            sha256(Path(__file__).parents[1] / name) == value for name, value in harness.items()
        ),
    )
    dump(args.output / "results.json", {"metadata": metadata, "rows": rows})
    dump(args.output / "metadata.json", metadata)
    if not metadata["package_unchanged"] or not metadata["harness_unchanged"]:
        raise RuntimeError("Measured package or harness changed during the campaign")
    if any(row["status"] == "harness_error" for row in rows):
        raise RuntimeError("Campaign contains harness failures; inspect retained rows")


if __name__ == "__main__":
    main()
