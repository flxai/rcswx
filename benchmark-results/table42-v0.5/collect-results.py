"""Retain compact measurements and exact hashes of the complete request evidence."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    spec = json.loads((ROOT / "campaign.json").read_text())
    targets_path = ROOT / "harness/tests/fixtures/table42/targets.json"
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
        "schema": "rcswx-table42-four-columns-observations-v1",
        "campaign_sha256": sha256(ROOT / "campaign.json"),
        "targets_sha256": sha256(targets_path),
        "runner_sha256": sha256(ROOT / "run-campaign.py"),
        "collector_sha256": sha256(Path(__file__)),
        "harness_archive_sha256": sha256(ROOT / "harness.tar.gz"),
        "memory_worker_sha256": sha256(ROOT / "harness/benchmarks/memory.py"),
        "source_request_sha256": sources,
        "commands": json.loads((ROOT / "commands.json").read_text()),
        "rows": rows,
    }
    (ROOT / "observations.json").write_text(json.dumps(evidence, indent=2) + "\n")
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
