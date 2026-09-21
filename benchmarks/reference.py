"""Compare the pinned source and native pipeline, keeping failures in the corpus."""

import argparse
import contextlib
import gc
import hashlib
import io
import json
import platform
import random
import resource
import statistics
import subprocess
import sys
import time
import tracemalloc
from functools import wraps
from pathlib import Path

import numpy as np
import psutil
import torch
from rcswx import (
    PCFG,
    Alignment,
    MatrixOperation,
    Reconstructor,
    _core,
    genotype,
    raw_crossover,
    recursive,
    sampling,
    select_operations,
    valid_combinations,
    validated_crossover,
)
from rcswx.grammars import einspace

from tests.reference.corpus import build_case
from tests.reference.original import (
    chain,
    environment,
    limits,
    load,
    operation_record,
    tree,
    tree_record,
)

WORKLOADS = (
    "ordered_tie_alignment",
    "unequal_tied_histories",
    "nested_routing_alignment",
    "branching_macro_alignment",
    "binary_reference_failure",
    "raw_ordered_crossover",
    "fractional_selector_failure",
    "dependency_distribution",
    "dependency_selection",
    "selection_draw",
    "validated_construction",
    "validated_retries",
)


def descriptions(name):
    first = chain(["linear(16)", "linear(32)"] * 8 + ["linear(4)"])
    second = chain(["linear(32)", "linear(16)"] * 8 + ["linear(4)"])
    if name in ("ordered_tie_alignment", "raw_ordered_crossover"):
        return first, second
    if name == "unequal_tied_histories":
        return chain(["identity"] * 16), chain(["identity"])
    if name == "nested_routing_alignment":
        return (
            "routing",
            "identity",
            ("routing", "identity", chain(["identity"]), "identity"),
            "identity",
        ), ("routing", "identity", chain(["relu"]), "identity")
    if name == "branching_macro_alignment":
        return tuple(
            (f"branching({arity})", f"clone({arity})", chain(["identity"]), f"add({arity})")
            for arity in (4, 8)
        )
    if name == "binary_reference_failure":
        return tuple(
            ("branching(2)", "clone(2)", chain([inner]), chain(["relu"]), "add(2)")
            for inner in ("identity", "relu")
        )
    if name == "fractional_selector_failure":
        return tuple(
            ("routing", before, chain(["identity"]), "identity")
            for before in ("perm(0,1,2)", "perm(0,2,1)")
        )
    raise ValueError(name)


def request(name, native):
    reference = load()
    if name in ("dependency_selection", "dependency_distribution", "selection_draw"):
        factory = MatrixOperation if native else reference.algorithm.MatrixOperation
        edits = [
            factory(op_id=index, value=np.float64(1), enabler_ops=[], disabler_ops=[])
            for index in range(12)
        ]
        for index, edit in enumerate(edits):
            edit.enabler_ops = [[] for _ in range(index)]
            edit.disabler_ops = [[other] for other in edits[:index]]
        if name == "dependency_distribution":
            if native:
                return lambda: list(zip(*valid_combinations(edits), strict=True))
            return lambda: list(reference.algorithm.combinations(edits).items())
        function = select_operations if native else reference.algorithm.select_operations
        if name == "selection_draw":
            choice = np.random.choice
            captured = []

            def capture(*args, **kwargs):
                captured.append((args, kwargs))
                return choice(*args, **kwargs)

            np.random.choice = capture
            try:
                function(edits)
            finally:
                np.random.choice = choice
            ((args, kwargs),) = captured
            return lambda: choice(*args, **kwargs)
        return lambda: function(edits)
    if name in ("validated_construction", "validated_retries"):
        parents = [
            build_case(
                (
                    "einspace",
                    ("sequential", ("computation", f"linear({width})"), ("computation", "relu")),
                    (2, 8),
                    "network",
                ),
                owned=native,
            )[0]
            for width in (16, 32)
        ]
        guard = parents[0].limiter
        if native:
            sampler = Reconstructor(PCFG(einspace.grammar, guard), guard, "iterative")
            rebuild = sampler.re_id
        else:
            sampler = reference.evolution.Evolver(
                pcfg=reference.pcfg.PCFG(reference.einspace.grammar, guard), limiter=guard
            )
            rebuild = sampler.re_id

        def constrained(root):
            rebuilt = rebuild(root)
            if name == "validated_retries" and rebuilt.output_params["shape"][-1] > 16:
                raise MemoryError("benchmark output-width limit")
            return rebuilt

        if native:
            return lambda: validated_crossover(
                *parents, rebuild=constrained, batch_shape=guard.batch_shape
            )
        sampler.re_id = constrained
        return lambda: sampler.recursive_constrained_smith_waterman_crossover(*parents)
    guard = limits()
    parents = [tree(description, limiter=guard) for description in descriptions(name)]
    if name in ("raw_ordered_crossover", "fractional_selector_failure"):
        function = (
            raw_crossover
            if native
            else reference.algorithm.recursive_constrained_smith_waterman_crossover
        )
    else:
        function = Alignment if native else reference.algorithm.AlignmentMatrixRecursive
    return lambda: function(*parents, limiter=guard)


def fingerprint(name, result):
    if name == "dependency_selection":
        value = [operation_record(op) for op in result]
    elif name == "dependency_distribution":
        value = [(mask, float(cost)) for mask, cost in result]
    elif name == "selection_draw":
        value = result
    elif name in ("validated_construction", "validated_retries"):
        root, report = result
        value = (
            tree_record(root),
            {
                key: [operation_record(op) for op in item]
                if key in ("crossover_operations", "crossover_all_operations")
                else item
                for key, item in report.items()
            },
        )
    elif name == "raw_ordered_crossover":
        root, selected, operations, *distances = result
        value = (
            tree_record(root),
            list(map(operation_record, selected)),
            list(map(operation_record, operations)),
            distances,
        )
    else:
        value = float(result.distance), list(map(operation_record, result.operations))
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


@contextlib.contextmanager
def profile_stages(native):
    """Separate diagnostic pass; primary timings never include instrumentation."""
    reference = load()
    totals, depths, restores = {}, {}, []
    decode_started = None
    previous_profile = sys.getprofile()

    def watch(owner, name, stage):
        function = getattr(owner, name)

        @wraps(function)
        def timed(*args, **kwargs):
            nonlocal decode_started
            if stage == "edit_translation" and decode_started is not None:
                seconds, calls = totals.get("native_path_decoding", (0.0, 0))
                totals["native_path_decoding"] = (
                    seconds + time.perf_counter() - decode_started,
                    calls + 1,
                )
                decode_started = None
            depths[stage] = depths.get(stage, 0) + 1
            start = time.perf_counter()
            completed = False
            try:
                result = function(*args, **kwargs)
                completed = True
                return result
            finally:
                depths[stage] -= 1
                if depths[stage] == 0:
                    seconds, calls = totals.get(stage, (0.0, 0))
                    totals[stage] = seconds + time.perf_counter() - start, calls + 1
                if completed and stage == "matrix_and_native_marshalling":
                    decode_started = time.perf_counter()

        restores.append((owner, name, function))
        setattr(owner, name, timed)

    alignment = Alignment if native else reference.algorithm.AlignmentMatrixRecursive
    watch(alignment, "breakdown", "input_tokenization")
    watch(alignment, "update_id", "parent_id_update")
    if native:
        watch(_core, "recursive_align", "matrix_and_native_marshalling")
        watch(sampling, "valid_combinations", "distribution_enumeration")
        watch(recursive, "select_operations", "distribution_and_selection")
    else:
        watch(alignment, "initialize_matrix", "matrix")
        watch(alignment, "calculate_matrix", "matrix")
        watch(reference.algorithm, "select_operations", "distribution_and_selection")
    watch(alignment, "calculate_restrictions", "edit_translation")
    watch(alignment, "generate_offspring", "edit_application")
    watch(
        Reconstructor if native else reference.sampler.Sampler, "re_id", "metadata_reconstruction"
    )
    watch(
        genotype.DerivationTreeNode if native else reference.state.DerivationTreeNode,
        "build",
        "model_construction",
    )
    watch(torch.nn.Module, "_call_impl", "validation_forward")
    if native:
        snapshot_code = next(
            value
            for value in Alignment.__init__.__code__.co_consts
            if getattr(value, "co_name", None) == "snapshot"
        )
        encoding_starts = []

        def encoding_profile(frame, event, argument):
            if previous_profile is not None:
                previous_profile(frame, event, argument)
            if frame.f_code is snapshot_code:
                if event == "call":
                    encoding_starts.append(time.perf_counter())
                elif event == "return":
                    seconds, calls = totals.get("native_token_encoding", (0.0, 0))
                    totals["native_token_encoding"] = (
                        seconds + time.perf_counter() - encoding_starts.pop(),
                        calls + 1,
                    )

        sys.setprofile(encoding_profile)
    try:
        yield totals
    finally:
        if native:
            sys.setprofile(previous_profile)
        for owner, name, function in reversed(restores):
            setattr(owner, name, function)


def seed_rng(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def measured_call(name, native, call):
    matrix_duration, native_stats = None, None
    start = time.perf_counter()
    try:
        result = call()
    except Exception as error:
        duration = time.perf_counter() - start
        outcome = {"status": "failure", "exception": type(error).__name__}
    else:
        duration = time.perf_counter() - start
        outcome = {"status": "success", "result_sha256": fingerprint(name, result)}
        if hasattr(result, "compute_time"):
            matrix_duration = result.compute_time
            paths = result.paths if native else result.matrix[-1][-1].paths
            outcome["retained_histories"] = len(paths)
            native_stats = getattr(result, "stats", None)
    numpy_state = np.random.get_state()
    outcome["python_state_sha256"] = hashlib.sha256(repr(random.getstate()).encode()).hexdigest()
    outcome["numpy_state_sha256"] = hashlib.sha256(
        numpy_state[1].tobytes() + repr(numpy_state[2:]).encode()
    ).hexdigest()
    outcome["torch_state_sha256"] = hashlib.sha256(
        torch.get_rng_state().numpy().tobytes()
    ).hexdigest()
    return duration, matrix_duration, outcome, native_stats


def worker(name, native, repeats):
    torch.set_num_threads(1)
    load()
    durations, matrix_durations, outcomes, stage_runs, native_stats = [], [], [], [], []
    with contextlib.redirect_stdout(io.StringIO()):
        # Warm framework dispatch without using the measured parents or RNG state.
        warm = request(name, native)
        seed_rng(42)
        try:
            warm()
        except Exception:
            pass
        del warm
        gc.collect()
        before_rss = psutil.Process().memory_info().rss
        before_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        for seed in range(repeats):
            call = request(name, native)
            seed_rng(seed)
            duration, matrix_duration, outcome, stats = measured_call(name, native, call)
            durations.append(duration)
            if matrix_duration is not None:
                matrix_durations.append(matrix_duration)
            if stats is not None:
                native_stats.append(stats)
            outcomes.append(outcome)
            del call
            gc.collect()

            with profile_stages(native) as stages:
                call = request(name, native)
                stages.clear()
                seed_rng(seed)
                duration, _, profiled_outcome, _ = measured_call(name, native, call)
            assert profiled_outcome == outcome, (name, outcome, profiled_outcome)
            if name in ("dependency_distribution", "dependency_selection", "selection_draw"):
                stage = {
                    "dependency_distribution": "distribution_enumeration",
                    "dependency_selection": "distribution_and_selection",
                    "selection_draw": "selection_draw",
                }[name]
                stages[stage] = duration, 1
            stage_runs.append(
                {
                    stage: {"milliseconds": seconds * 1000, "calls": calls}
                    for stage, (seconds, calls) in stages.items()
                }
            )
            del call
            gc.collect()
        call = request(name, native)
        seed_rng(42)
        tracemalloc.start()
        try:
            call()
        except Exception:
            pass
        _, python_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        del call
        gc.collect()
    return {
        "name": name,
        "backend": "native" if native else "original",
        "repeats": repeats,
        "median_ms": statistics.median(durations) * 1000,
        "matrix_median_ms": statistics.median(matrix_durations) * 1000
        if matrix_durations
        else None,
        "diagnostic_stage_runs": stage_runs,
        "native_stats": native_stats,
        "python_traced_peak_bytes": python_peak,
        "rss_before_bytes": before_rss,
        "rss_after_gc_bytes": psutil.Process().memory_info().rss,
        "rss_peak_before_bytes": before_peak,
        "rss_peak_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "outcomes": outcomes,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=WORKLOADS)
    parser.add_argument("--native", action="store_true")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(args.worker, args.native, args.repeats), allow_nan=False))
        return
    torch.set_num_threads(1)
    rows = []
    problems = []
    for name in WORKLOADS:
        pair = []
        for native in (False, True):
            command = [
                sys.executable,
                "-m",
                "benchmarks.reference",
                "--worker",
                name,
                "--repeats",
                str(args.repeats),
            ]
            if native:
                command.append("--native")
            try:
                child = subprocess.run(
                    command,
                    check=True,
                    text=True,
                    capture_output=True,
                    timeout=args.timeout,
                    cwd=Path(__file__).resolve().parents[1],
                )
            except subprocess.TimeoutExpired:
                row = {
                    "name": name,
                    "backend": "native" if native else "original",
                    "status": "measurement_timeout",
                    "timeout_seconds": args.timeout,
                }
            except subprocess.CalledProcessError as error:
                row = {
                    "name": name,
                    "backend": "native" if native else "original",
                    "status": "worker_failure",
                    "exit_status": error.returncode,
                    "stderr": error.stderr[-3000:],
                }
                problems.append(f"{name}: {row['backend']} worker failure")
            else:
                row = json.loads(child.stdout)
            pair.append(row)
        if all("outcomes" in row for row in pair):
            if pair[0]["outcomes"] != pair[1]["outcomes"]:
                problems.append(f"{name}: semantic mismatch")
        rows.extend(pair)
        print(
            json.dumps(
                {
                    "workload": name,
                    "backends": [row.get("status", "completed") for row in pair],
                }
            ),
            file=sys.stderr,
            flush=True,
        )
    payload = {
        "corpus": "pinned-reference-recursive-v1",
        "environment": environment(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "native_extension": {
            "path": _core.__file__,
            "sha256": hashlib.sha256(Path(_core.__file__).read_bytes()).hexdigest(),
        },
        "native_sources": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                Path("crates/rcswx-core/src/recursive.rs"),
                Path("crates/rcswx-core/src/selection.rs"),
                Path("python/rcswx/recursive.py"),
                Path("python/rcswx/sampling.py"),
                Path("python/rcswx/pipeline.py"),
            )
        },
        "rows": rows,
        "problems": problems,
        "memory_scope": "Each backend/workload runs in a fresh process. RSS is total process high-water including framework imports and warmup; tracemalloc measures Python allocations only, not native heap. No retained cross-request alignment cache.",
        "semantic_scope": "Every completed pair must match offspring/witness, reference failures, retained-history count, and Python/NumPy/Torch post-call RNG states. Measurement timeouts remain explicit rows, not unsupported inputs or algorithmic retry limits.",
        "timing_scope": "Primary totals are uninstrumented and exclude fixture setup/imports. Diagnostic stages use separate identically seeded requests and verify identical outcomes/RNG states; they include tracing/wrapper overhead, overlap, and must not be summed. Tokenization, parent-ID update, Python token-record encoding, native matrix/PyO3 marshalling, and Python path decoding are reported separately. The reference directly uses Python node objects and has no native encoding/decoding phase. Distribution enumeration and the NumPy draw also have isolated workloads. The draw reuses arguments captured from the actual selector only in this benchmark; production has no distribution cache. Validated-retry work enforces the same output-width limit in both backends.",
    }
    serialized = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(serialized)
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "workloads": len(WORKLOADS),
                    "completed": sum("outcomes" in row for row in rows),
                    "timeouts": sum(row.get("status") == "measurement_timeout" for row in rows),
                    "problems": problems,
                }
            )
        )
    else:
        print(serialized, end="")
    if problems:
        raise SystemExit(
            "Benchmark failed; all completed and failed rows were retained in the output"
        )


if __name__ == "__main__":
    main()
