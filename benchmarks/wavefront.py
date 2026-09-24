"""Measure one public edit_path call; keep fingerprints outside its timed window.

Run with the Python from an explicitly installed baseline or candidate wheel:
python -m benchmarks.wavefront --label baseline --baseline --output target/baseline.json
The chain, routing and branching layouts extend the repository's portable fixtures.
"""

import argparse
import gc
import hashlib
import json
import os
import platform
import statistics
import subprocess
import time
from pathlib import Path

from benchmarks.memory import resident


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
        if family == "branch":
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
        choices=["chain", "wrapper", "branch"],
        default=["chain", "wrapper", "branch"],
    )
    parser.add_argument("--collapse", choices=["both", "on", "off"], default="both")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    if args.repeats < 1 or any(size < 2 for size in args.sizes):
        parser.error("repeats must be positive and sizes at least two")
    import rcswx
    from rcswx import _core, edit_path

    report = {
        "schema": 1,
        "label": args.label,
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "package": rcswx.__file__,
        "extension_sha256": hashlib.sha256(Path(_core.__file__).read_bytes()).hexdigest(),
        "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "profiled": args.profile,
        "timing": "One synchronous public edit_path; preparation and plan assembly included; input construction, fingerprinting, and destruction excluded. First call recorded separately.",
        "rows": [],
    }
    flags = [False, True] if args.collapse == "both" else [args.collapse == "on"]
    for family in args.families:
        for size in args.sizes:
            first, second = parents(family, size)
            hashes = [
                hashlib.sha256(value.to_json().encode()).hexdigest() for value in (first, second)
            ]
            for collapse in flags:
                for workers in args.workers:
                    row = {
                        "family": family,
                        "size": size,
                        "collapse_corners": collapse,
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
                    if not args.baseline:
                        options["workers"] = workers
                    for repeat in range(args.repeats + 1):
                        gc.collect()
                        reset = True
                        try:
                            Path("/proc/self/clear_refs").write_text("5\n")
                        except OSError:
                            reset = False
                        before = resident()
                        started = time.perf_counter_ns()
                        try:
                            plan = edit_path(first, second, **options)
                        except Exception as error:
                            row["error"] = {
                                "type": type(error).__name__,
                                "message": str(error),
                                "seconds": (time.perf_counter_ns() - started) / 1e9,
                            }
                            break
                        elapsed = (time.perf_counter_ns() - started) / 1e9
                        after = resident()
                        paths = plan._native.paths_json()
                        sample = {
                            "seconds": elapsed,
                            "rss_peak_bytes": after["VmHWM"] if reset else None,
                            "baseline_rss_bytes": before["VmRSS"],
                            "stats": plan.stats,
                            "execution": getattr(plan, "execution", None),
                            "distance": plan.distance,
                            "paths_sha256": hashlib.sha256(paths.encode()).hexdigest(),
                            "tokens": [
                                len(plan._prepared.first_tokens()),
                                len(plan._prepared.second_tokens()),
                            ],
                        }
                        if args.profile:
                            sample["timings"] = plan.timings
                        if repeat == 0:
                            row["cold"] = sample
                        else:
                            row["samples"].append(sample)
                        del plan
                    if row["samples"]:
                        row["median_seconds"] = statistics.median(
                            sample["seconds"] for sample in row["samples"]
                        )
                        fingerprints = {
                            sample["paths_sha256"] for sample in [row["cold"], *row["samples"]]
                        }
                        if len(fingerprints) != 1:
                            raise RuntimeError("Repeated requests changed ordered path content")
                    report["rows"].append(row)
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    args.output.write_text(json.dumps(report, indent=2) + "\n")
                    print(
                        json.dumps(
                            {
                                key: row[key]
                                for key in (
                                    "family",
                                    "size",
                                    "collapse_corners",
                                    "workers",
                                    "median_seconds",
                                    "error",
                                )
                                if key in row
                            }
                        ),
                        flush=True,
                    )


if __name__ == "__main__":
    main()
