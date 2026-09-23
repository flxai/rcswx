import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from benchmarks.memory import comparisons
from benchmarks.table42_campaign import (
    CampaignBudget,
    configuration_key,
    configuration_statuses,
    next_attempt,
    primary_configurations,
)

CASES = ("resnet18-mixer8", "resnet18-mixer12", "resnet34-mixer8", "resnet34-mixer12")


def observation(configuration, *, repeat=0, status="complete", returned=True):
    return {
        **configuration,
        "phase": "align",
        "measure": "rss",
        "repeat": repeat,
        "reverse": False,
        "runtime_config_sha256": "same-profile",
        "stratum": "paired_current_reference",
        "status": status,
        "semantic_verification": "full" if status == "complete" else "unverified",
        "worker_seconds": 1.0,
        "worker_limits": {"timeout_seconds": 120, "rss_limit_mib": 1024},
        "measurement": {"seconds": 0.5, "align_wall_seconds": 0.5, "profiled": False},
        "outcome": {
            "status": "returned" if returned else "raised",
            "digest": "same-semantic-observation",
            "details": {"distance": 2.0},
        },
        "identity": {"host": "test-host", "affinity": [0], "source_identity": {"revision": "same"}},
    }


def test_pilots_cover_every_pair_mode_and_backend_before_repeating():
    configurations = primary_configurations(CASES)
    selected = {configuration_key(row) for row in configurations}
    rows, visited = [], []
    for _ in configurations:
        plan = next_attempt(configurations, rows, selected, 3, 1800)
        assert plan["tier"] == "pilot"
        assert plan["timeout_seconds"] == 120
        visited.append(configuration_key(plan))
        rows.append(
            observation({key: plan[key] for key in ("case", "collapse_corners", "backend")})
        )
    assert len(set(visited)) == 16
    next_plan = next_attempt(configurations, rows, selected, 3, 1800)
    assert next_plan["repeat"] == 1
    assert next_plan["backend"] == "native"


def test_mode_selection_does_not_erase_unrun_modes_or_repeat_before_their_pilots():
    configurations = primary_configurations(CASES)
    selected = {configuration_key(row) for row in configurations if not row["collapse_corners"]}
    rows = [observation(row) for row in configurations if configuration_key(row) in selected]
    assert next_attempt(configurations, rows, selected, 3, 1800) is None
    states = configuration_statuses(configurations, rows, 3, 100, selected)
    assert len(states) == 16
    assert {row["status"] for row in states if row["collapse_corners"]} == {"not_run"}
    exhausted = configuration_statuses(configurations, rows, 3, 0, selected)
    assert {row["status"] for row in exhausted if row["collapse_corners"]} == {"budget_unrun"}


def test_timeout_gets_only_one_deliberate_larger_tier_retry():
    configurations = primary_configurations(CASES)
    rows = [observation(row) for row in configurations]
    rows[0] = observation(configurations[0], status="timeout")
    selected = {configuration_key(configurations[0])}
    assert next_attempt(configurations, rows, selected, 3, 120) is None
    retry = next_attempt(configurations, rows, selected, 3, 1800)
    assert retry["tier"] == "extended"
    row = observation(configurations[0], repeat=retry["repeat"], status="timeout")
    row["worker_limits"]["timeout_seconds"] = 1800
    rows.append(row)
    assert next_attempt(configurations, rows, selected, 3, 1800) is None


@pytest.mark.parametrize("status", ["rss_limit", "construction_error", "fingerprint_error"])
def test_failed_configurations_are_not_blindly_repeated(status):
    configurations = primary_configurations(CASES)
    rows = [observation(row) for row in configurations]
    rows[0] = observation(configurations[0], status=status)
    assert (
        next_attempt(configurations, rows, {configuration_key(configurations[0])}, 3, 1800) is None
    )


def test_budget_survives_an_actual_controller_exit_and_accounts_its_reservation(tmp_path):
    path = tmp_path / "budget.json"
    source = """import os, sys
from benchmarks.table42_campaign import CampaignBudget
request = {'case':'case','phase':'align','measure':'rss','backend':'reference','repeat':0,'reverse':False,'collapse_corners':False}
with CampaignBudget(sys.argv[1], {'fixture':'same'}, 10) as budget:
    budget.reserve(request, 4)
    os._exit(0)
"""
    subprocess.run([sys.executable, "-c", source, str(path)], check=True)
    with CampaignBudget(path, {"fixture": "same"}, 10) as budget:
        assert budget.remaining == 6
        assert budget.data["rows"][0]["status"] == "controller_interrupted"
        assert budget.data["rows"][0]["worker_seconds"] is None
        assert budget.data["rows"][0]["budget_charge_seconds"] == 4
        budget.reserve({"case": "another"}, 3)
        budget.finish({"worker_seconds": 1.25})
    with CampaignBudget(path, {"fixture": "same"}, 10) as budget:
        assert budget.remaining == 4.75
        with pytest.raises(ValueError):
            budget.reserve({}, 5)


def test_active_campaign_and_live_worker_cannot_be_duplicated(tmp_path):
    path = tmp_path / "budget.json"
    with CampaignBudget(path, {}, 10) as budget:
        with pytest.raises(BlockingIOError):
            with CampaignBudget(path, {}, 10):
                pass
        budget.reserve({}, 2)
        budget.attach_worker(os.getpid(), tmp_path)
    with pytest.raises(RuntimeError):
        with CampaignBudget(path, {}, 10):
            pass
    assert json.loads(path.read_text())["spent_seconds"] == 0


def test_changed_source_identity_and_budget_cannot_reset_existing_accounting(tmp_path):
    path = tmp_path / "budget.json"
    with CampaignBudget(path, {"source": "original"}, 10):
        pass
    with pytest.raises(ValueError):
        with CampaignBudget(path, {"source": "changed"}, 10):
            pass
    with pytest.raises(ValueError):
        with CampaignBudget(path, {"source": "original"}, 20):
            pass


def test_failed_atomic_checkpoint_does_not_replace_previous_budget(tmp_path, monkeypatch):
    path = tmp_path / "budget.json"
    with CampaignBudget(path, {}, 10) as budget:
        original_replace = Path.replace

        def fail_checkpoint(source, destination):
            if Path(destination) == path:
                raise OSError("synthetic filesystem failure")
            return original_replace(source, destination)

        with monkeypatch.context() as scope:
            scope.setattr(Path, "replace", fail_checkpoint)
            with pytest.raises(OSError):
                budget.reserve({}, 4)
    with CampaignBudget(path, {}, 10) as budget:
        assert budget.remaining == 10
        budget.reserve({}, 10)
        budget.finish({"worker_seconds": 2})
    with CampaignBudget(path, {}, 10) as budget:
        assert budget.remaining == 8


@pytest.mark.parametrize(
    "difference", ["mode", "profile", "host", "fingerprint", "exception", "witness"]
)
def test_only_equivalent_complete_same_configuration_returns_are_speedup_eligible(difference):
    configuration = primary_configurations(CASES)[0]
    reference = observation(configuration)
    native = observation({**configuration, "backend": "native"})
    assert comparisons([reference, native])[0]["speedup_eligible"]
    changed = copy.deepcopy(native)
    if difference == "mode":
        changed["collapse_corners"] = True
    elif difference == "profile":
        changed["runtime_config_sha256"] = "different-profile"
    elif difference == "host":
        changed["identity"]["host"] = "another-host"
    elif difference == "fingerprint":
        changed["status"] = "timeout"
        changed["partial"] = {"stage": "fingerprinting", "scalar_outcome": {"distance": 2.0}}
    elif difference == "exception":
        reference["outcome"]["status"] = changed["outcome"]["status"] = "raised"
    else:
        changed["outcome"]["digest"] = "same-distance-different-ordered-history"
    assert not any(row["speedup_eligible"] for row in comparisons([reference, changed]))


def test_duplicate_repetitions_cannot_silently_overwrite_evidence():
    row = observation(primary_configurations(CASES)[0])
    with pytest.raises(ValueError):
        comparisons([row, copy.deepcopy(row)])
