"""Persistent bounded scheduling around benchmarks.memory's measurement engine."""

import argparse
import fcntl
import fnmatch
import json
import math
import os
import platform
import time
from pathlib import Path

from .memory import comparisons, digest, dump, run_worker


class CampaignBudget:
    """One locked ledger; interrupted reservations are charged conservatively."""

    def __init__(self, path, identity, limit_seconds):
        if not math.isfinite(limit_seconds) or limit_seconds <= 0:
            raise ValueError("Campaign budget must be finite and positive")
        self.path = Path(path)
        self.identity = identity
        self.limit_seconds = limit_seconds
        self.lock = None
        self.data = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = self.path.with_name(self.path.name + ".lock").open("a+")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if self.path.exists():
                self.data = json.loads(self.path.read_text())
                if self.data.get("schema") != "rcswx-table42-budget-v1":
                    raise ValueError("Unknown campaign ledger schema; refusing to reset it")
                if (
                    self.data["identity"] != self.identity
                    or self.data["limit_seconds"] != self.limit_seconds
                ):
                    raise ValueError(
                        "Campaign identity or total budget changed; refusing to reuse/reset ledger"
                    )
                if not math.isfinite(self.data["spent_seconds"]) or self.data["spent_seconds"] < 0:
                    raise ValueError("Invalid cumulative campaign accounting")
            else:
                self.data = {
                    "schema": "rcswx-table42-budget-v1",
                    "identity": self.identity,
                    "limit_seconds": self.limit_seconds,
                    "spent_seconds": 0.0,
                    "pending": None,
                    "rows": [],
                    "invocations": [],
                }
            self.recover_interrupted()
            self.persist()
            return self
        except BaseException:
            self.lock.close()
            self.lock = None
            raise

    def __exit__(self, *error):
        if self.lock is not None:
            self.lock.close()
            self.lock = None

    @property
    def remaining(self):
        reserved = self.data["pending"]["reserved_seconds"] if self.data["pending"] else 0
        return max(0.0, self.limit_seconds - self.data["spent_seconds"] - reserved)

    def persist(self):
        dump(self.path, self.data)

    @staticmethod
    def process_alive(pid, created):
        if pid is None:
            return False
        import psutil

        try:
            process = psutil.Process(pid)
            return process.is_running() and (created is None or process.create_time() == created)
        except psutil.NoSuchProcess:
            return False

    def recover_interrupted(self):
        pending = self.data["pending"]
        if pending is None:
            return
        if self.process_alive(pending.get("worker_pid"), pending.get("worker_created")):
            raise RuntimeError(
                "A reserved measurement worker is still alive; stop its owning job before resuming"
            )
        if self.process_alive(pending.get("controller_pid"), pending.get("controller_created")):
            raise RuntimeError("The previous campaign controller is still alive")
        row = dict(pending["request"])
        partial = None
        if pending.get("work_dir"):
            work = Path(pending["work_dir"])
            for filename in ("result.json", "progress.json"):
                if (work / filename).exists():
                    partial = json.loads((work / filename).read_text())
                    break
        row.update(
            status="controller_interrupted",
            returncode=None,
            worker_seconds=None,
            worker_wall_seconds=None,
            budget_charge_seconds=pending["reserved_seconds"],
            budget_charge_rule="Full reserved timeout; actual whole-worker elapsed time is unavailable",
            interrupted_stage=(partial or {}).get("stage", "unknown"),
            partial=partial,
            work_dir=pending.get("work_dir"),
            semantic_verification="unverified",
        )
        self.data["rows"].append(row)
        self.data["spent_seconds"] += pending["reserved_seconds"]
        self.data["pending"] = None

    def reserve(self, request, timeout):
        import psutil

        if self.data["pending"] is not None:
            raise RuntimeError("A campaign reservation is already active")
        if not math.isfinite(timeout) or timeout <= 0 or timeout > self.remaining:
            raise ValueError("Worker reservation exceeds remaining campaign budget")
        self.data["pending"] = {
            "request": request,
            "reserved_seconds": timeout,
            "controller_pid": os.getpid(),
            "controller_created": psutil.Process().create_time(),
            "worker_pid": None,
            "worker_created": None,
            "work_dir": None,
        }
        self.persist()

    def attach_worker(self, pid, work):
        import psutil

        self.data["pending"].update(worker_pid=pid, work_dir=str(work))
        try:
            self.data["pending"]["worker_created"] = psutil.Process(pid).create_time()
        except psutil.NoSuchProcess:
            pass
        self.persist()

    def finish(self, row):
        if self.data["pending"] is None:
            raise RuntimeError("Cannot settle a worker without its budget reservation")
        seconds = row["worker_seconds"]
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("Actual worker wall time must be finite and nonnegative")
        row["budget_charge_seconds"] = seconds
        row["budget_charge_rule"] = (
            "Observed complete worker wall time, including setup/fingerprinting"
        )
        self.data["rows"].append(row)
        self.data["spent_seconds"] += seconds
        self.data["pending"] = None
        self.persist()


def configuration_key(row):
    return row["case"], row["collapse_corners"], row["backend"]


def successful(row):
    return (
        row["status"] == "complete"
        and row.get("outcome", {}).get("status") == "returned"
        and row.get("semantic_verification") == "full"
        and isinstance(row["outcome"].get("digest"), str)
        and type(row["outcome"].get("details", {}).get("distance")) in (int, float)
        and math.isfinite(row["outcome"]["details"]["distance"])
    )


def primary_configurations(cases):
    return [
        {"case": case, "collapse_corners": collapse, "backend": backend}
        for case in cases
        for collapse in (False, True)
        for backend in ("reference", "native")
    ]


def next_attempt(configurations, rows, selected_keys, successes, timeout):
    histories = {configuration_key(config): [] for config in configurations}
    for row in rows:
        histories[configuration_key(row)].append(row)
    missing = [config for config in configurations if not histories[configuration_key(config)]]
    if missing:
        for config in missing:
            if configuration_key(config) in selected_keys:
                return {
                    **config,
                    "repeat": 0,
                    "tier": "pilot",
                    "timeout_seconds": min(120.0, timeout),
                }
        return None  # Cover the other requested modes before repeating these easy cases.
    mismatched = {
        (row["case"], row["collapse_corners"])
        for row in comparisons(rows)
        if row["status"] == "mismatch"
    }
    candidates = []
    for order, config in enumerate(configurations):
        key = configuration_key(config)
        previous = histories[key]
        if key not in selected_keys or (key[0], key[1]) in mismatched:
            continue
        if sum(successful(row) for row in previous) >= successes:
            continue
        last = previous[-1]
        repeat = max(row["repeat"] for row in previous) + 1
        if successful(last):
            tier = "repeat"
        elif (
            last["status"] == "timeout"
            and len(previous) == 1
            and timeout > last["worker_limits"]["timeout_seconds"]
        ):
            tier = "extended"
        elif last["status"] == "controller_interrupted" and len(previous) == 1:
            tier = "resumed"
        else:
            continue  # Keep RSS, construction, semantic and repeated-timeout failures visible.
        backend_order = ("reference", "native") if repeat % 2 == 0 else ("native", "reference")
        candidates.append(
            (
                repeat,
                order // 2,
                backend_order.index(config["backend"]),
                {
                    **config,
                    "repeat": repeat,
                    "tier": tier,
                    "timeout_seconds": min(1800.0, timeout),
                },
            )
        )
    return min(candidates, key=lambda item: item[:3])[3] if candidates else None


def configuration_statuses(configurations, rows, requested_successes, remaining, selected_keys):
    result = []
    for configuration in configurations:
        key = configuration_key(configuration)
        attempts = [row for row in rows if configuration_key(row) == key]
        count = sum(successful(row) for row in attempts)
        if not attempts:
            status = "budget_unrun" if remaining <= 0 else "not_run"
        elif count >= requested_successes:
            status = "complete"
        elif not successful(attempts[-1]):
            status = attempts[-1]["status"]
        else:
            status = "repetitions_incomplete"
        result.append(
            {
                **configuration,
                "status": status,
                "attempts": len(attempts),
                "successful_repetitions": count,
                "requested_successful_repetitions": requested_successes,
                "selected_in_invocation": key in selected_keys,
            }
        )
    return result


def execution_identity(config, config_path, rss_limit):
    import rcswx
    import torch
    from rcswx import _core

    from tests.reference.original import environment
    from tests.reference.table42 import sha256

    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[1]
    package = Path(rcswx.__file__).parent
    harness = [
        "benchmarks/memory.py",
        "benchmarks/table42_campaign.py",
        "tests/reference/table42.py",
        "tests/reference/original.py",
        "tests/reference/networks.py",
        "tests/reference/corpus.py",
        "tests/test_reference_construction.py",
    ]
    return {
        "host": platform.node(),
        "environment": environment(),
        "affinity_cpu": config["affinity_cpu"],
        "runtime_config_sha256": sha256(config_path),
        "stratum": config["stratum"],
        "native_sha256": sha256(_core.__file__),
        "native_extension": str(_core.__file__),
        "package_sha256": {
            str(path.relative_to(package)): sha256(path) for path in sorted(package.rglob("*.py"))
        },
        "harness_sha256": {relative: sha256(root / relative) for relative in harness},
        "source_identity": config["reference_environment"],
        "rss_limit_mib": rss_limit,
        "boundary": "complete_alignment_object_return",
    }


def controller(args):
    from tests.reference.table42 import load_runtime_config, table_pairs

    config = load_runtime_config(args.runtime_config)
    cases = table_pairs(config)
    if args.list:
        for case in cases.values():
            print(f"{case.name:24} {case.left} -> {case.right} shape={case.shape}")
        return 0
    if args.output is None:
        raise SystemExit("--output is required unless --list is used")
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    captures = (args.captures_dir or args.output.with_suffix(".captures")).resolve()
    captures.mkdir(parents=True, exist_ok=True)
    configurations = primary_configurations(cases)
    selected_modes = (
        (False, True) if args.collapse_corners == "both" else (args.collapse_corners == "on",)
    )
    selected_keys = {
        configuration_key(row)
        for row in configurations
        if row["collapse_corners"] in selected_modes
        and (
            not args.case or any(fnmatch.fnmatchcase(row["case"], pattern) for pattern in args.case)
        )
    }
    if not selected_keys:
        raise SystemExit("No primary table configurations match the selection")
    identity = execution_identity(config, args.runtime_config, args.rss_limit_mib)
    with CampaignBudget(args.budget_file, identity, args.wall_budget_seconds) as ledger:
        if args.output.exists():
            existing = json.loads(args.output.read_text())
            if existing.get("campaign_identity_sha256") != digest(identity):
                raise ValueError("Refusing to overwrite an output belonging to another campaign")
        invocation = {
            "started_unix_seconds": time.time(),
            "output": str(args.output),
            "timeout_seconds": args.timeout,
            "requested_successes": args.repeats,
            "collapse_corners": args.collapse_corners,
            "case_filters": args.case,
        }
        ledger.data["invocations"].append(invocation)
        ledger.persist()
        report = {
            "schema": "rcswx-table42-campaign-v1",
            "suite": "table42",
            **identity,
            "campaign_identity_sha256": digest(identity),
            "runtime_config": config,
            "cases": [case.record() for case in cases.values()],
            "protocol": {
                "seed": config["seed"],
                "threads": config["threads"],
                "affinity_cpu": config["affinity_cpu"],
                "collapse_modes": [False, True],
                "fixed_corner_threshold": 0.25,
                "boundary": "complete_alignment_object_return",
                "fresh_process_per_call": True,
                "warmup_calls": 0,
                "concurrent_workers": 1,
                "pilot_worker_limit_seconds": 120,
                "extended_worker_limit_seconds": 1800,
                "requested_successful_repetitions": args.repeats,
                "one_run_results_are_pilots": True,
                "rss_limit_mib": args.rss_limit_mib,
                "rss": "Linux high-water reset after setup; absolute window peak includes the baseline and excludes fingerprinting. Failed reset yields null window metrics.",
                "after_release": "Diagnostic after fingerprinting and releasing returned objects; prepared inputs remain live.",
                "whole_worker": "Budget and supervisor bounds include imports, setup, alignment, fingerprinting and release.",
                "stdout": "Both backends use per-worker log files; unmodified source debug formatting/output remains inside the call boundary.",
                "resume": "Run under an owning process group/cgroup with whole-job cleanup. Live reserved workers prevent resume; unknown elapsed time consumes the full reservation.",
            },
            "budget_file": str(args.budget_file.resolve()),
            "invocations": ledger.data["invocations"],
        }

        def checkpoint():
            report.update(
                rows=ledger.data["rows"],
                comparisons=comparisons(ledger.data["rows"]),
                configurations=configuration_statuses(
                    configurations,
                    ledger.data["rows"],
                    args.repeats,
                    ledger.remaining,
                    selected_keys,
                ),
                budget={
                    "limit_seconds": ledger.limit_seconds,
                    "spent_seconds": ledger.data["spent_seconds"],
                    "remaining_seconds": ledger.remaining,
                    "pending": ledger.data["pending"],
                    "overrun_seconds": max(
                        0.0, ledger.data["spent_seconds"] - ledger.limit_seconds
                    ),
                },
            )
            dump(args.output, report)

        checkpoint()  # All sixteen records exist before the first subprocess.
        while ledger.remaining > 0:
            plan = next_attempt(
                configurations, ledger.data["rows"], selected_keys, args.repeats, args.timeout
            )
            if plan is None:
                break
            requested_timeout = plan.pop("timeout_seconds")
            actual_timeout = min(requested_timeout, ledger.remaining)
            worker_args = argparse.Namespace(**vars(args))
            worker_args.collapse_corners = "on" if plan["collapse_corners"] else "off"
            worker_args.timeout = actual_timeout
            reservation = {
                **plan,
                "phase": "align",
                "measure": "rss",
                "reverse": False,
                "stratum": config["stratum"],
                "runtime_config_sha256": identity["runtime_config_sha256"],
                "worker_limits": {
                    "timeout_seconds": actual_timeout,
                    "rss_limit_mib": args.rss_limit_mib,
                },
                "budget_truncated_timeout": actual_timeout < requested_timeout,
            }
            ledger.reserve(reservation, actual_timeout)
            checkpoint()
            try:
                row = run_worker(
                    worker_args,
                    cases[plan["case"]],
                    "align",
                    "rss",
                    plan["backend"],
                    plan["repeat"],
                    captures,
                    on_spawn=ledger.attach_worker,
                )
                row.update(
                    tier=plan["tier"], budget_truncated_timeout=actual_timeout < requested_timeout
                )
                ledger.finish(row)
                checkpoint()
                print(
                    f"{plan['case']} collapse={worker_args.collapse_corners} {plan['backend']} attempt={plan['repeat']} {row['status']} stage={row.get('interrupted_stage', row.get('stage'))}; budget={ledger.data['spent_seconds']:.1f}/{ledger.limit_seconds:.1f}s",
                    flush=True,
                )
            except BaseException:
                checkpoint()
                raise
        report["primary_pilot_coverage_complete"] = all(
            row["attempts"] for row in report["configurations"]
        )
        report["invocation_finished"] = True
        report["termination"] = (
            "budget_exhausted" if ledger.remaining <= 0 else "eligible_selection_exhausted"
        )
        checkpoint()
    return int(
        any(row["status"] == "mismatch" for row in report["comparisons"])
        or any(row["status"] in {"worker_error", "fingerprint_error"} for row in report["rows"])
    )
