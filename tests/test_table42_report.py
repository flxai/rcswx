import copy
import json
import sys
from pathlib import Path

import pytest

from benchmarks.table42_report import main, summarize
from tests.test_table42_campaign import CASES, observation, primary_configurations

TARGETS = json.loads((Path(__file__).parent / "fixtures/table42/targets.json").read_text())


def report_for(rows):
    return {
        "rows": rows,
        "campaign_identity_sha256": "test-campaign",
        "runtime_config": {},
        "budget": {},
        "configurations": [],
    }


def measured(backend, repeat, seconds, mode=False):
    config = {"case": CASES[0], "backend": backend, "collapse_corners": mode}
    row = observation(config, repeat=repeat)
    row["measurement"]["align_wall_seconds"] = seconds
    return row


def test_censored_and_unmatched_samples_do_not_contaminate_a_paired_ratio():
    reference = measured("reference", 0, 4)
    native = measured("native", 0, 2)
    censored = measured("reference", 1, 0.01)
    censored.update(status="timeout", semantic_verification="unverified")
    unpaired = measured("native", 1, 100)
    summary = summarize(report_for([reference, native, censored, unpaired]), TARGETS)
    row = summary["modes"]["off"][0]
    assert row["speedup"] == 2
    assert row["matched_repetitions"] == [0]
    assert row["backends"]["reference"]["metrics"]["align_wall_seconds"]["values"] == [4]
    assert row["backends"]["native"]["excluded_repetitions"] == [1]
    assert summary["modes"]["on"][0]["speedup"] is None
    assert {
        (item["backend"], item["repeat"])
        for item in summary["historical_audit"]
        if item["paired_semantics_verified"]
    } == {("reference", 0), ("native", 0)}


def test_a_semantic_mismatch_cannot_be_hidden_by_selecting_agreeing_repetitions():
    rows = [
        measured(backend, repeat, 4 if backend == "reference" else 2)
        for repeat in (0, 1)
        for backend in ("reference", "native")
    ]
    rows[-1]["outcome"]["digest"] = "different-ordered-history"
    summary = summarize(report_for(rows), TARGETS)
    assert summary["modes"]["off"][0]["speedup"] is None
    assert len(summary["failures_and_exclusions"]) == 2
    assert not any(item["paired_semantics_verified"] for item in summary["historical_audit"])


def test_partial_scalar_is_preserved_without_promoting_it_to_verified_performance():
    row = measured("reference", 0, 12)
    row.update(status="timeout", semantic_verification="unverified")
    del row["outcome"]
    row["partial"] = {
        "stage": "fingerprinting",
        "scalar_outcome": {"status": "returned", "distance": 60.375},
    }
    summary = summarize(report_for([row]), TARGETS)
    assert summary["modes"]["off"][0]["backends"]["reference"]["distances"] == [60.375]
    audit = summary["historical_audit"][0]
    assert audit["two_decimal_rendering"] == "60.38"
    assert audit["printed_format_agrees"] is True
    assert audit["paired_semantics_verified"] is False
    assert summary["modes"]["off"][0]["speedup"] is None


def test_historical_targets_remain_literal_and_unobserved_configurations_remain_visible():
    targets = copy.deepcopy(TARGETS)
    summary = summarize(report_for([]), targets)
    assert targets == TARGETS
    assert (
        summary["historical_targets"]["ordered_pairs"][-1]["printed_compute_time"]
        == "741.57 minutes"
    )
    assert [row["case"] for row in summary["modes"]["off"]] == list(CASES)
    assert [row["case"] for row in summary["modes"]["on"]] == list(CASES)
    assert all(row["speedup"] is None for rows in summary["modes"].values() for row in rows)


def test_nonfinite_distance_is_preserved_without_becoming_a_successful_ratio():
    rows = [observation(config) for config in primary_configurations(CASES)[:2]]
    for row in rows:
        row["outcome"]["details"]["distance"] = {"nonfinite": "+inf"}
    summary = summarize(report_for(rows), TARGETS)
    assert summary["modes"]["off"][0]["speedup"] is None
    assert summary["historical_audit"][0]["observed_distance"] == {"nonfinite": "+inf"}
    assert summary["historical_audit"][0]["printed_format_agrees"] is None


@pytest.mark.parametrize("mismatch", ["runtime_audit", "historical_target"])
def test_report_rejects_stale_audits_and_rewritten_historical_targets(
    tmp_path, monkeypatch, mismatch
):
    report = report_for([])
    report["runtime_config_sha256"] = "measured-configuration"
    source = tmp_path / "campaign.json"
    source.write_text(json.dumps(report))
    for symbol in TARGETS["models"]:
        (tmp_path / f"audit-{symbol}.json").write_text(
            json.dumps({"runtime_config_sha256": "another-configuration"})
        )
    targets = copy.deepcopy(TARGETS)
    if mismatch == "historical_target":
        targets["ordered_pairs"][-1]["printed_compute_time"] = "741.57 seconds"
    target_file = tmp_path / "targets.json"
    target_file.write_text(json.dumps(targets))
    output = tmp_path / "output"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "table42_report",
            str(source),
            "--targets",
            str(target_file),
            "--output-dir",
            str(output),
        ],
    )
    with pytest.raises(ValueError):
        main()
    assert not (output / "REPORT.md").exists()


def test_failed_rss_reset_cannot_mix_different_repetition_sets_in_memory_ratio():
    rows = [
        measured(backend, repeat, 4 if backend == "reference" else 2)
        for repeat in (0, 1)
        for backend in ("reference", "native")
    ]
    for row, peak in zip(rows, (800, 400, None, 200), strict=True):
        row["measurement"]["rss_peak_bytes"] = peak * 1024**2 if peak else None
        row["measurement"]["rss_peak_reset_error"] = None if peak else "Permission denied"
    result = summarize(report_for(rows), TARGETS)["modes"]["off"][0]
    assert result["matched_repetitions"] == [0, 1]
    assert result["memory_ratio"]["matched_repetitions"] == [0]
    assert result["memory_ratio"]["python_over_rust"] == 2
