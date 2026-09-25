"""Measure one public edit_path call; keep fingerprints outside its timed window.

Run with the Python from an explicitly installed baseline or candidate wheel:
python -m benchmarks.wavefront --label baseline --baseline --output target/baseline.json
The chain, routing and branching layouts extend the repository's portable fixtures.
"""

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from importlib.metadata import distributions
from pathlib import Path

from benchmarks.memory import resident


def installed_artifact(module):
    for distribution in distributions(path=[str(Path(module.__file__).parent.parent)]):
        if distribution.metadata["Name"] == "rcswx":
            source = distribution.read_text("direct_url.json")
            return json.loads(source) if source else None
    return None


def sequence(items):
    if len(items) == 1:
        return items[0]
    middle = len(items) // 2
    return ("sequential", sequence(items[:middle]), sequence(items[middle:]))


def parents(family, size):
    from rcswx import Architecture

    def build(changed):
        leaves = []
        for index in range(size):
            parameter = 16 + 2 * index + int(changed and index == size // 2)
            item = ("computation", (f"linear({parameter})",))
            if family == "wrapper":
                item = ("routing", ("identity",), item, ("identity",))
            leaves.append(item)
        if family == "nested":
            tree = leaves[-1]
            for leaf in reversed(leaves[:-1]):
                tree = ("routing", ("identity",), ("sequential", leaf, tree), ("identity",))
        elif family == "branch":
            middle = len(leaves) // 2
            tree = (
                "branching(2)",
                ("clone(2)",),
                sequence(leaves[:middle]),
                sequence(leaves[middle:]),
                ("add(2)",),
            )
        else:
            tree = sequence(leaves)
        return Architecture.from_tree(tree)

    return build(False), build(True)


def measured(module, pair, options, profiled):
    gc.collect()
    reset = True
    try:
        Path("/proc/self/clear_refs").write_text("5\n")
    except OSError:
        reset = False
    before = resident()
    cpu = time.process_time_ns()
    started = time.perf_counter_ns()
    try:
        plan = module.edit_path(*pair, **options)
    except Exception as error:
        return {
            "error": {
                "type": type(error).__name__,
                "message": str(error),
                "seconds": (time.perf_counter_ns() - started) / 1e9,
            }
        }
    elapsed = (time.perf_counter_ns() - started) / 1e9
    cpu = (time.process_time_ns() - cpu) / 1e9
    after = resident()
    sample = {
        "seconds": elapsed,
        "process_cpu_seconds": cpu,
        "rss_peak_bytes": after["VmHWM"] if reset else None,
        "baseline_rss_bytes": before["VmRSS"],
        "stats": plan.stats,
        "execution": getattr(plan, "execution", None),
        "distance": plan.distance,
        "paths_sha256": hashlib.sha256(plan._native.paths_json().encode()).hexdigest(),
        "tokens": [len(plan._prepared.first_tokens()), len(plan._prepared.second_tokens())],
    }
    if profiled:
        sample["timings"] = plan.timings
    return sample


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument(
        "--baseline", action="store_true", help="Do not pass the not-yet-existing workers keyword"
    )
    parser.add_argument("--workers", type=int, nargs="+", default=[1])
    parser.add_argument("--sizes", type=int, nargs="+", default=[32, 64, 128])
    parser.add_argument(
        "--families",
        nargs="+",
        choices=["chain", "wrapper", "nested", "branch"],
        default=["chain", "wrapper", "branch"],
    )
    parser.add_argument("--collapse", choices=["both", "on", "off"], default="both")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument(
        "--compare-package",
        type=Path,
        help="Alternate calls with an independently installed original package directory",
    )
    args = parser.parse_args()
    if args.repeats < 1 or any(size < 2 for size in args.sizes):
        parser.error("repeats must be positive and sizes at least two")
    import rcswx
    from rcswx import _core

    report = {
        "schema": 1,
        "label": args.label,
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "package": rcswx.__file__,
        "installation": installed_artifact(rcswx),
        "extension_sha256": hashlib.sha256(Path(_core.__file__).read_bytes()).hexdigest(),
        "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "profiled": args.profile,
        "timing": "One synchronous public edit_path; preparation and plan assembly included; input construction, fingerprinting, and destruction excluded. First call recorded separately.",
        "rows": [],
    }
    original = None
    if args.compare_package:
        path = args.compare_package.resolve()
        spec = importlib.util.spec_from_file_location(
            "_rcswx_wavefront_baseline",
            path / "__init__.py",
            submodule_search_locations=[str(path)],
        )
        original = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = original
        spec.loader.exec_module(original)
        report["baseline_package"] = original.__file__
        report["baseline_installation"] = installed_artifact(original)
        report["baseline_extension_sha256"] = hashlib.sha256(
            Path(original._core.__file__).read_bytes()
        ).hexdigest()
        report["comparison"] = (
            "Separate installed packages/native libraries in one interpreter; mode order alternates forward/reverse and rotates between repetitions. Use standalone runs for isolated RSS."
        )
    flags = [False, True] if args.collapse == "both" else [args.collapse == "on"]
    for family in args.families:
        for size in args.sizes:
            pair = parents(family, size)
            hashes = [hashlib.sha256(value.to_json().encode()).hexdigest() for value in pair]
            original_pair = (
                tuple(original.Architecture.from_json(value.to_json()) for value in pair)
                if original
                else None
            )
            for collapse in flags:
                variants = [
                    (f"workers={workers}", rcswx, pair, workers, args.baseline)
                    for workers in args.workers
                ]
                if original:
                    variants.insert(0, ("original", original, original_pair, 1, True))
                runs = []
                for variant, module, inputs, workers, legacy in variants:
                    row = {
                        "family": family,
                        "size": size,
                        "collapse_corners": collapse,
                        "variant": variant,
                        "workers": workers,
                        "input_sha256": hashes,
                        "samples": [],
                    }
                    options = {
                        "collapse_corners": collapse,
                        "profile": args.profile,
                        "limits": {
                            "max_work": 5_000_000,
                            "max_output": 1_000_000,
                            "max_allocation_bytes": 512 * 1024 * 1024,
                        },
                    }
                    if not legacy:
                        options["workers"] = workers
                    runs.append((module, inputs, options, row))
                for repeat in range(args.repeats + 1):
                    # Rotate order rather than comparing long back-to-back runs:
                    # a hybrid laptop's thermal/frequency drift is substantial.
                    ordered = runs if repeat % 2 == 0 else runs[::-1]
                    offset = (repeat // 2) % len(ordered)
                    for module, inputs, options, row in ordered[offset:] + ordered[:offset]:
                        if "error" in row:
                            continue
                        sample = measured(module, inputs, options, args.profile)
                        if "error" in sample:
                            row.update(sample)
                        elif repeat == 0:
                            row["cold"] = sample
                        else:
                            row["samples"].append(sample)
                fingerprints = set()
                for _, _, _, row in runs:
                    if row["samples"]:
                        row["median_seconds"] = statistics.median(
                            sample["seconds"] for sample in row["samples"]
                        )
                        fingerprints.update(
                            (
                                sample["distance"],
                                sample["paths_sha256"],
                                json.dumps(sample["stats"], sort_keys=True),
                            )
                            for sample in [row["cold"], *row["samples"]]
                        )
                    report["rows"].append(row)
                    print(
                        json.dumps(
                            {
                                key: row[key]
                                for key in (
                                    "family",
                                    "size",
                                    "collapse_corners",
                                    "variant",
                                    "workers",
                                    "median_seconds",
                                    "error",
                                )
                                if key in row
                            }
                        ),
                        flush=True,
                    )
                if len(fingerprints) > 1:
                    raise RuntimeError(
                        "Compared requests changed distance, ordered histories, or canonical statistics"
                    )
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
