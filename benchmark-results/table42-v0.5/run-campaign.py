"""Run exact tagged wheels through the existing Table 4.2 measurement worker."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent


def request(spec, version, case_name, repeat):
    from benchmarks.consolidation import package_snapshot
    from benchmarks.memory import dump, run_worker
    from tests.reference.table42 import load_runtime_config, table_pairs

    os.sched_setaffinity(0, {spec["serial_cpu"]})
    release = spec["releases"][version]
    package = package_snapshot()
    assert package["extension_sha256"] == release["artifact"]["extension_sha256"]
    config_path = ROOT / f"{version}-runtime.json"
    config = load_runtime_config(config_path)
    args = SimpleNamespace(
        suite="table42",
        runtime_config=config_path,
        collapse_corners="on",
        reverse=False,
        timeout=1800,
        rss_limit_mib=16384,
        workers=release["workers"],
        affinity_cpus=release["cpus"],
    )
    captures = ROOT / "captures" / version
    captures.mkdir(parents=True, exist_ok=True)
    row = run_worker(
        args, table_pairs(config)[case_name], "align", "rss", "native", repeat, captures
    )
    row.update(
        release=version,
        package=package,
        package_unchanged=package_snapshot() == package,
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        request_runner_argv=sys.argv,
    )
    dump(ROOT / "requests" / f"{case_name}-{version}-{repeat}.json", row)
    assert row["package_unchanged"], "Package changed during measurement"
    assert row["status"] == "complete", row.get("diagnostic", row["status"])
    assert row["outcome"]["status"] == "returned"
    assert row["identity"]["affinity"] == release["cpus"]
    if version == "v0.5":
        execution = row["measurement"]["native_execution"]
        assert execution["requested_workers"] == execution["worker_limit"] == 10
        assert execution["pool_capacity"] == 10 and execution["peak_jobs"] <= 10
    print(
        json.dumps(
            {
                "case": case_name,
                "version": version,
                "repeat": repeat,
                "seconds": row["measurement"]["align_wall_seconds"],
                "distance": row["outcome"]["details"]["distance"],
            }
        ),
        flush=True,
    )


def idle(cpus):
    import psutil

    began = time.monotonic()
    while True:
        usage = psutil.cpu_percent(interval=1, percpu=True)
        available = psutil.virtual_memory().available
        if max(usage[cpu] for cpu in cpus) < 10 and available > 24 * 1024**3:
            return {"cpu_percent": usage, "available_memory": available}
        if time.monotonic() - began > 120:
            raise RuntimeError(f"Selected CPUs/memory remain busy: {usage}, {available}")
        time.sleep(2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", nargs=3, metavar=("VERSION", "CASE", "REPEAT"))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    spec = json.loads((ROOT / "campaign.json").read_text())
    if args.request:
        version, case_name, repeat = args.request
        request(spec, version, case_name, int(repeat))
        return
    os.sched_setaffinity(0, {spec["controller_cpu"]})
    (ROOT / "requests").mkdir(exist_ok=True)
    records = []
    for version, release in spec["releases"].items():
        env = {**spec["environment"], "PYTHONPATH": f"{release['site']}:{spec['harness']}"}
        path = ROOT / f"{version}-runtime.json"
        command = [
            spec["python"],
            "-m",
            "tests.reference.table42",
            "--resolve-config",
            str(path),
            "--admit-supplied-factories",
            "--cpu",
            str(spec["serial_cpu"]),
        ]
        # Configuration resolution validates its own affinity before model setup.
        command = ["taskset", "-c", str(spec["serial_cpu"]), *command]
        result = subprocess.run(
            command, env={**os.environ, **env}, capture_output=True, text=True, timeout=120
        )
        records.append(
            {
                "kind": "resolve",
                "version": version,
                "command": command,
                "environment": env,
                "returncode": result.returncode,
                "stderr": result.stderr,
            }
        )
        if result.returncode:
            raise RuntimeError(result.stderr)
    cases = spec["cases"][:1] if args.smoke else spec["cases"]
    repeats = [-1] if args.smoke else range(3)
    for case_name in cases:
        for repeat in repeats:
            versions = list(spec["releases"])
            shift = repeat % len(versions)
            versions = versions[shift:] + versions[:shift]
            for version in versions:
                destination = ROOT / "requests" / f"{case_name}-{version}-{repeat}.json"
                if destination.exists():
                    raise RuntimeError(f"Refusing to overwrite {destination}")
                preflight = idle(spec["parallel_cpus"])
                release = spec["releases"][version]
                env = {**spec["environment"], "PYTHONPATH": f"{release['site']}:{spec['harness']}"}
                command = [
                    spec["python"],
                    str(Path(__file__)),
                    "--request",
                    version,
                    case_name,
                    str(repeat),
                ]
                result = subprocess.run(
                    command, env={**os.environ, **env}, capture_output=True, text=True, timeout=1860
                )
                record = {
                    "kind": "request",
                    "case": case_name,
                    "version": version,
                    "repeat": repeat,
                    "command": command,
                    "environment": env,
                    "preflight": preflight,
                    "returncode": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                }
                records.append(record)
                (ROOT / ("smoke-commands.json" if args.smoke else "commands.json")).write_text(
                    json.dumps(records, indent=2) + "\n"
                )
                print(result.stdout, end="", flush=True)
                if result.returncode:
                    raise RuntimeError(result.stderr)
    print("Requested table measurements completed.", flush=True)


if __name__ == "__main__":
    main()
