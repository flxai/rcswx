"""Run the pinned local sequence corpus; never discard unsupported/exhausted cases."""

import argparse
import hashlib
import json
import platform
import resource
import statistics
import sys
import time
import tracemalloc
from pathlib import Path

import rcswx
import torch
from rcswx.policies import SAMPLING_VERSION
from torch import nn


def widths(width, edits):
    dimensions = [4] + [width] * edits + [4]
    return nn.Sequential(*(nn.Linear(a, b) for a, b in zip(dimensions, dimensions[1:])))


def cases():
    yield "nine_width_edits", widths(8, 9), widths(16, 9), rcswx.Limits()
    yield (
        "shape_rejection_k3",
        nn.Identity(),
        nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 4)),
        rcswx.Limits(max_attempts=3),
    )
    yield (
        "normalization",
        nn.Sequential(nn.Linear(4, 8), nn.BatchNorm1d(8), nn.Dropout(0.25), nn.Linear(8, 4)),
        nn.Sequential(nn.Linear(4, 16), nn.BatchNorm1d(16), nn.Dropout(0.25), nn.Linear(16, 4)),
        rcswx.Limits(),
    )
    yield "twenty_edit_bound", widths(8, 20), widths(16, 20), rcswx.Limits()


class Residual(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(4, 4)

    def forward(self, x):
        return self.layer(x) + x


def run(repeats):
    torch.manual_seed(0)
    torch.set_num_threads(1)
    spec = rcswx.TensorSpec((2, 4))
    result = {
        "corpus": "ordered-local-v1",
        "recipe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "machine": platform.machine(),
            "platform": sys.platform,
            "torch_threads": torch.get_num_threads(),
            "rcswx": rcswx.__version__,
            "sampling": SAMPLING_VERSION,
        },
        "repeats": repeats,
        "seeds": list(range(repeats)),
        "warmup_seed": 42,
        "strata": [],
        "admission_failures": [],
    }
    try:
        rcswx.prepare(Residual(), input_spec=spec)
    except rcswx.UnsupportedArchitecture as error:
        result["admission_failures"].append({"name": "residual", "error": type(error).__name__})
    else:
        raise AssertionError("the postponed residual domain was unexpectedly admitted")
    for name, left, right, limits in cases():
        a, b = (rcswx.prepare(model, input_spec=spec) for model in (left, right))
        path = rcswx.edit_path(a, b)
        # Warm constructors/framework dispatch before recording stage timings.
        rcswx.crossover(a, b, path=path, seed=42, limits=limits)
        stages = {
            stage: []
            for stage in (
                "capture_seconds",
                "distance_seconds",
                "alignment_seconds",
                "plan_export_seconds",
                "recapture_seconds",
                "distribution_seconds",
                "proposal_validation_seconds",
                "materialization_seconds",
                "total_seconds",
            )
        }
        outcomes = []
        accepted = novel = 0
        attempted = 0
        for seed in range(repeats):
            outcome = {"seed": seed}
            start = time.perf_counter()
            captured = tuple(rcswx.prepare(model, input_spec=spec) for model in (left, right))
            stages["capture_seconds"].append(time.perf_counter() - start)
            del captured
            start = time.perf_counter()
            assert rcswx.distance(a, b) == path.distance
            stages["distance_seconds"].append(time.perf_counter() - start)
            start = time.perf_counter()
            fresh_path = rcswx.edit_path(a, b)
            stages["alignment_seconds"].append(time.perf_counter() - start)
            del fresh_path
            start = time.perf_counter()
            exported_plan = rcswx.EditPath._create(a, b, path.native)
            stages["plan_export_seconds"].append(time.perf_counter() - start)
            del exported_plan
            try:
                value = rcswx.crossover_with_report(a, b, path=path, seed=seed, limits=limits)
            except rcswx.SamplingExhausted as error:
                report = error.report
                outcome["status"] = "exhausted"
            else:
                report = value.report
                outcome.update(status="accepted", mask=report["selected_mask"])
                accepted += 1
                value.model.eval()
                assert tuple(value.model(torch.ones(2, 4)).shape) == spec.shape
                start = time.perf_counter()
                key = rcswx.prepare(value.model, input_spec=spec).architecture_key
                stages["recapture_seconds"].append(time.perf_counter() - start)
                outcome["novel"] = key not in (a.architecture_key, b.architecture_key)
                novel += outcome["novel"]
            outcome.update(attempts=report["attempts"], rejections=report["rejections"])
            outcomes.append(outcome)
            assert report["alignment_computations"] == 0
            attempted += report["attempts"]
            for stage in (
                "distribution_seconds",
                "proposal_validation_seconds",
                "materialization_seconds",
                "total_seconds",
            ):
                stages[stage].append(report["timings"][stage])
        tracemalloc.start()
        allocation_probe = rcswx.crossover_with_report(a, b, path=path, seed=42, limits=limits)
        _, python_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        result["strata"].append(
            {
                "name": name,
                "parent_nodes": [len(a.operations), len(b.operations)],
                "edits": len(path.edits),
                "cost_ticks": path.cost_ticks,
                "accepted": accepted,
                "exhausted": repeats - accepted,
                "proposals": attempted,
                "novel": novel,
                "outcomes": outcomes,
                "median_ms": {
                    key.removesuffix("_seconds"): 1000 * statistics.median(values)
                    if values
                    else None
                    for key, values in stages.items()
                },
                "alignment": dict(path.native.stats),
                "proposal": allocation_probe.report["proposal"],
                "python_request_peak_bytes": python_peak,
            }
        )
    result["process_peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result["memory_scope"] = (
        "Native capacity charges and traced Python request allocations are not total RSS; RSS is process high-water including PyTorch. No shared global cache."
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=32)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 1 <= args.repeats <= 256:
        parser.error("repeats must be in [1,256]")
    value = run(args.repeats)
    payload = json.dumps(value, indent=2, allow_nan=False) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.write_text(payload)
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "strata": [
                        {
                            key: row[key]
                            for key in ("name", "accepted", "exhausted", "novel", "median_ms")
                        }
                        for row in value["strata"]
                    ],
                    "process_peak_rss_kib": value["process_peak_rss_kib"],
                }
            )
        )
