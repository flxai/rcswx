"""Evidence gates must not turn censored or different work into speedups."""

from copy import deepcopy

import pytest

from benchmarks.consolidation import (
    SCHEMA,
    compare_pair,
    compare_reports,
    probability_agreement,
    select_pairs,
)
from benchmarks.consolidation_pool import ASSIGNMENT, assigned_cpu


def observation(**changes):
    return {
        "status": "ok",
        "stage": "complete",
        "outcome_digest": "same-result",
        "outcome": {"value": 1.0},
        "phase": "distance",
        "cpu_seconds": 2.0,
        "profiled": False,
        "_campaign_complete": True,
        **changes,
    }


def test_only_complete_same_work_primary_calls_receive_speedups():
    first = observation()
    second = observation(cpu_seconds=1.0)
    name = "baseline-reference_vs_consolidated-reference"
    assert compare_pair(first, second, name, "distance")["cpu_speedup"] == 2.0

    for changed in (
        observation(outcome_digest="different-result"),
        observation(status="exception"),
        observation(profiled=True),
        observation(_campaign_complete=False),
        observation(outcome={"value": 0.0}),
    ):
        comparison = compare_pair(first, changed, name, "distance")
        assert comparison["speedup_eligible"] is False
        assert "cpu_speedup" not in comparison

    censored = observation(status="timeout", stage="measuring", outcome_digest=None)
    comparison = compare_pair(censored, censored, name, "distance")
    assert comparison["semantic_match"] is None
    assert comparison["speedup_eligible"] is False


def test_numerical_tolerance_requires_identical_distribution_inputs():
    reference = {
        "masks_digest": "masks",
        "costs_digest": "costs",
        "mask_count": 2,
        "operations_digest": "operations",
        "probabilities": [0.25, 0.75],
    }
    rounded = dict(reference, probabilities=[0.25 + 1e-13, 0.75 - 1e-13])
    assert probability_agreement(reference, rounded)["match"] is True
    assert probability_agreement(reference, dict(rounded, costs_digest="other"))["match"] is False
    assert (
        probability_agreement(reference, dict(reference, probabilities=[0.5, 0.5]))["match"]
        is False
    )
    assert (
        probability_agreement(reference, dict(reference, probabilities=[0.25, 0.8]))["match"]
        is False
    )


def campaign_metadata():
    metadata = {
        "schema": SCHEMA,
        "complete": True,
        "package_unchanged": True,
        "harness_unchanged": True,
        "corpus": "frozen",
        "identity": {"host": "ono"},
        "selection_sha256": None,
        "worker_limits": {},
        "harness": {},
        "profiled": False,
        "skewness": 0,
        "seed_policy": "recorded pair seed",
        "sampler_seeds": None,
    }
    return metadata


def test_native_natural_outcomes_are_not_paired_reference_speedups():
    metadata = campaign_metadata()
    reports = []
    for generation, sampler, child in (
        ("baseline", "reference", "reference-child"),
        ("consolidated", "native", "native-child"),
    ):
        row = observation(
            phase="raw",
            generation=generation,
            sampler=sampler,
            pair="pair",
            pair_sha256="parents",
            repeat_id=0,
            sampler_seed=12,
            control_digest=None,
            wall_seconds=2.0,
            outcome={"value": {"child": child}},
        )
        reports.append(
            {
                "metadata": dict(metadata, generation=generation, package={"digest": generation}),
                "rows": [row],
            }
        )
    result = compare_reports(reports)
    assert result["summaries"] == []
    assert {row["sampler"] for row in result["natural_outcomes"]} == {"native", "reference"}
    assert all(row["semantic_match"] is None for row in result["comparisons"])

    changed = deepcopy(reports[1])
    changed["metadata"]["identity"]["host"] = "different-host"
    assert len(compare_reports([reports[0], changed])["cohorts"]) == 2


def same_work_report(generation, execution=None):
    metadata = dict(campaign_metadata(), generation=generation, package={"digest": generation})
    row = observation(
        generation=generation,
        sampler="none",
        pair="pair",
        pair_sha256="parents",
        repeat_id=0,
        sampler_seed=12,
        control_digest=None,
    )
    if execution is not None:
        metadata["execution"] = execution
        row["worker_cpu"] = assigned_cpu("pair", "distance", 0, 12, execution["worker_cpus"])
    return {"metadata": metadata, "rows": [row]}


def parallel_execution(cpus):
    return {
        "mode": "parallel",
        "workers": len(cpus),
        "worker_cpus": cpus,
        "controller_cpu": 31,
        "assignment": ASSIGNMENT,
    }


def test_serial_and_parallel_work_cannot_be_paired_for_speedups():
    first = same_work_report("baseline")
    second = same_work_report("consolidated")
    assert compare_reports([first, second])["summaries"]
    second = same_work_report("consolidated", parallel_execution([0, 2]))
    result = compare_reports([first, second])
    assert len(result["cohorts"]) == 2
    assert not any(row["speedup_eligible"] for row in result["comparisons"])


def test_parallel_core_assignment_policy_separates_measurement_cohorts():
    first = same_work_report("baseline", parallel_execution([0, 2]))
    second = same_work_report("consolidated", parallel_execution([0, 2]))
    assert compare_reports([first, second])["summaries"]
    second = same_work_report("consolidated", parallel_execution([16, 17]))
    result = compare_reports([first, second])
    assert len(result["cohorts"]) == 2
    assert not any(row["speedup_eligible"] for row in result["comparisons"])


def test_parallel_report_rejects_a_request_measured_on_the_wrong_cpu():
    report = same_work_report("baseline", parallel_execution([0, 2]))
    report["rows"][0]["worker_cpu"] = 2 - report["rows"][0]["worker_cpu"]
    with pytest.raises(ValueError):
        compare_reports([report])


def test_same_result_on_different_cpus_is_not_a_paired_speedup():
    comparison = compare_pair(
        observation(worker_cpu=0),
        observation(worker_cpu=2),
        "baseline-reference_vs_consolidated-reference",
        "distance",
    )
    assert comparison["semantic_match"] is True
    assert comparison["speedup_eligible"] is False
    assert "cpu_speedup" not in comparison


def test_diagnostic_selection_keeps_distinct_engineering_strata():
    pairs = [{"id": str(index), "sha256": str(index), "nodes": index + 1} for index in range(100)]
    rows = [
        {
            "pair": pair["id"],
            "nodes": pair["nodes"],
            "backend": "native",
            "phase": phase,
            "status": "ok",
            "cpu_seconds": float((pair["nodes"] * 13) % 101),
        }
        for phase in ("distance", "raw")
        for pair in pairs
    ]
    rows.append(
        {"pair": "97", "nodes": 98, "backend": "native", "phase": "raw", "status": "timeout"}
    )
    comparisons = [
        dict(row, speedup=1 / (row["cpu_seconds"] + 1)) for row in rows if row["status"] == "ok"
    ]
    manifest = {"pairs": pairs}
    historical = {"rows": rows, "comparisons": comparisons}
    features = {"95": {"repeated_sibling_signatures": 1}, "96": {"branched_nodes": 1}}
    selection = select_pairs(manifest, historical, features, count=40)
    assert len({pair["id"] for pair in selection}) == 40
    assert selection == select_pairs(manifest, historical, features, count=40)
    reasons = {reason for pair in selection for reason in pair["reasons"]}
    assert {
        "resource_limited",
        "tied_history_candidate_repeated_siblings",
        "branched_structure",
    } <= reasons
