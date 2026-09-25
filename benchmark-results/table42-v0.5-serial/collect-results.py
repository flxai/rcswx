"""Retain compact measurements and exact hashes of the complete request evidence."""

import csv
import hashlib
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    spec = json.loads((ROOT / "campaign.json").read_text())
    harness = Path(spec["harness"])
    targets_path = harness / "tests/fixtures/table42/targets.json"
    targets = json.loads(targets_path.read_text())
    rows, sources = [], {}
    for target in targets["ordered_pairs"]:
        case = target["pair_id"]
        for version, release in spec["releases"].items():
            for repeat in range(3):
                path = ROOT / "requests" / f"{case}-{version}-{repeat}.json"
                raw = json.loads(path.read_text())
                assert raw["case"] == case and raw["release"] == version and raw["repeat"] == repeat
                assert raw["status"] == "complete" and raw["semantic_verification"] == "full"
                assert raw["outcome"]["status"] == "returned" and raw["package_unchanged"]
                assert raw["nodes"] == target["printed_node_counts"]
                assert raw["collapse_corners"] is True
                assert raw["identity"]["affinity"] == release["cpus"]
                assert raw["package"]["extension_sha256"] == release["artifact"]["extension_sha256"]
                assert (
                    raw["identity"]["source_identity"]["native_extension_sha256"]
                    == release["artifact"]["extension_sha256"]
                )
                assert raw["runner_sha256"] == sha256(ROOT / "run-campaign.py")
                assert f"{raw['outcome']['details']['distance']:.2f}" == target["printed_distance"]
                if version == "v0.5":
                    execution = raw["measurement"]["native_execution"]
                    assert execution["parallel_capable"] is True
                    assert execution["requested_workers"] == execution["worker_limit"] == 1
                    assert execution["pool_capacity"] is None
                    assert all(
                        execution[field] == 0
                        for field in (
                            "parallel_cells",
                            "parallel_rounds",
                            "jobs",
                            "peak_jobs",
                            "scratch_peak_bytes",
                            "narrow_fallbacks",
                            "alias_fallbacks",
                            "scratch_fallbacks",
                        )
                    )
                sources[str(path.relative_to(ROOT))] = sha256(path)
                rows.append(
                    {
                        "case": case,
                        "release": version,
                        "repeat": repeat,
                        "status": raw["status"],
                        "nodes": raw["nodes"],
                        "affinity": raw["identity"]["affinity"],
                        "measurement": raw["measurement"],
                        "outcome": raw["outcome"],
                        "fixture_runtime_sha256": raw["identity"]["fixture_manifest"][
                            "runtime_sha256"
                        ],
                        "source_identity": raw["identity"]["source_identity"],
                        "worker_limits": raw["worker_limits"],
                        "request_sha256": sources[str(path.relative_to(ROOT))],
                    }
                )
        matched = [row for row in rows if row["case"] == case]
        assert len({row["outcome"]["digest"] for row in matched}) == 1, f"Outcome mismatch: {case}"
        assert len({row["fixture_runtime_sha256"] for row in matched}) == 1, (
            f"Fixture mismatch: {case}"
        )
    evidence = {
        "schema": "rcswx-table42-serial-control-observations-v1",
        "campaign_sha256": sha256(ROOT / "campaign.json"),
        "targets_sha256": sha256(targets_path),
        "runner_sha256": sha256(ROOT / "run-campaign.py"),
        "collector_sha256": sha256(Path(__file__)),
        "harness_archive_sha256": sha256(harness.parent / "harness.tar.gz"),
        "memory_worker_sha256": sha256(harness / "benchmarks/memory.py"),
        "source_request_sha256": sources,
        "commands": json.loads((ROOT / "commands.json").read_text()),
        "rows": rows,
    }
    (ROOT / "observations.json").write_text(json.dumps(evidence, indent=2) + "\n")
    comparisons = []
    for case in spec["cases"]:
        samples = [row for row in rows if row["case"] == case]
        series = {}
        for version in spec["releases"]:
            measurements = [
                row["measurement"]
                for row in sorted(samples, key=lambda row: row["repeat"])
                if row["release"] == version
            ]
            seconds = [measurement["seconds"] for measurement in measurements]
            series[version] = {
                "seconds": seconds,
                "median_seconds": statistics.median(seconds),
                "min_seconds": min(seconds),
                "max_seconds": max(seconds),
                "median_peak_rss_mib": statistics.median(
                    measurement["rss_peak_bytes"] / 1024**2 for measurement in measurements
                ),
            }
        ratio = series["v0.5"]["median_seconds"] / series["v0.3"]["median_seconds"]
        comparisons.append(
            {
                "case": case,
                "nodes": samples[0]["nodes"],
                "distance": samples[0]["outcome"]["details"]["distance"],
                "outcome_digest": samples[0]["outcome"]["digest"],
                "series": series,
                "median_seconds_ratio_v05_over_v03": ratio,
                "change_percent": (ratio - 1) * 100,
                "paired_ratios_v05_over_v03": [
                    current / previous
                    for previous, current in zip(
                        series["v0.3"]["seconds"], series["v0.5"]["seconds"], strict=True
                    )
                ],
            }
        )
    summary = {
        "schema": "rcswx-table42-serial-control-summary-v1",
        "observations_sha256": sha256(ROOT / "observations.json"),
        "repeats_per_release": 3,
        "affinity": spec["releases"]["v0.5"]["cpus"],
        "workers": 1,
        "v05_parallel_capable": True,
        "v05_pool_created": False,
        "all_native_outcomes_equal": True,
        "comparisons": comparisons,
    }
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (ROOT / "comparison.csv").open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            [
                "case",
                "v0.3_serial_seconds",
                "v0.5_serial_seconds",
                "v0.5_over_v0.3",
                "change_percent",
                "v0.3_peak_rss_mib",
                "v0.5_peak_rss_mib",
            ]
        )
        for comparison in comparisons:
            series = comparison["series"]
            writer.writerow(
                [
                    comparison["case"],
                    series["v0.3"]["median_seconds"],
                    series["v0.5"]["median_seconds"],
                    comparison["median_seconds_ratio_v05_over_v03"],
                    comparison["change_percent"],
                    series["v0.3"]["median_peak_rss_mib"],
                    series["v0.5"]["median_peak_rss_mib"],
                ]
            )
    print(
        json.dumps(
            {
                "rows": len(rows),
                "all_native_outcomes_equal": True,
                "all_published_distances_matched_when_rounded": True,
                "cases": spec["cases"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
