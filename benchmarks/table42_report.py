"""Render observed Table 4.2 results without replacing the historical targets."""

import argparse
import csv
import html
import json
import math
import statistics
from collections import Counter
from pathlib import Path

from tests.reference.table42 import verify_assets

from .memory import comparisons, dump

ROOT = Path(__file__).resolve().parents[1]
METRICS = (
    "align_wall_seconds",
    "baseline_rss_bytes",
    "rss_peak_bytes",
    "rss_peak_delta_bytes",
    "retained_rss_bytes",
    "after_release_rss_bytes",
)


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def stats(values):
    values = [value for value in values if finite(value)]
    return {
        "values": values,
        "n": len(values),
        "median": statistics.median(values) if values else None,
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def scalar(row):
    outcome = row.get("outcome", {})
    if outcome.get("status") == "returned":
        return outcome.get("details", {}).get("distance")
    partial = row.get("scalar_outcome") or (row.get("partial") or {}).get("scalar_outcome") or {}
    return partial.get("distance") if partial.get("status") == "returned" else None


def unique(values):
    result, seen = [], set()
    for value in values:
        key = json.dumps(value, sort_keys=True)
        if value is not None and key not in seen:
            seen.add(key)
            result.append(value)
    return result


def same_component(pair, section, name):
    values = [row.get("outcome", {}) for row in pair.values()]
    if section:
        values = [value.get(section, {}) for value in values]
    return (
        len(values) == 2
        and all(name in value for value in values)
        and values[0][name] == values[1][name]
    )


def summarize(report, targets):
    rows = report["rows"]
    verified = comparisons(rows)
    modes, sensitivity, historical_audit = {}, [], []
    for mode in (False, True):
        mode_rows = []
        for target in targets["ordered_pairs"]:
            case = target["pair_id"]
            selected = [
                row for row in rows if row["case"] == case and row["collapse_corners"] == mode
            ]
            paired = {}
            for row in selected:
                paired.setdefault(row["repeat"], {})[row["backend"]] = row
            matches = []
            verification = []
            for comparison in verified:
                if comparison["case"] != case or comparison["collapse_corners"] != mode:
                    continue
                pair = paired[comparison["repeat"]]
                if comparison["speedup_eligible"] and all(
                    finite(row.get("measurement", {}).get("align_wall_seconds"))
                    and row["measurement"]["align_wall_seconds"] > 0
                    for row in pair.values()
                ):
                    matches.append(comparison["repeat"])
                verification.append(
                    {
                        **comparison,
                        "components": {
                            key: same_component(pair, section, field)
                            if comparison["semantics_equal"] is not None
                            else None
                            for key, section, field in (
                                ("distance", "details", "distance"),
                                ("ordered_histories", "details", "histories_digest"),
                                ("edits_and_dependencies", "details", "operations_digest"),
                                ("parents_before", None, "parents_before_digest"),
                                ("parents_after", None, "parents_after_digest"),
                                ("rng_before", None, "rng_before"),
                                ("rng_after", None, "rng_after"),
                            )
                        },
                    }
                )
            if any(item["status"] == "mismatch" for item in verification):
                matches = []  # Never select only the agreeing repetitions of a failing configuration.
            backends = {}
            for backend in ("reference", "native"):
                attempts = [row for row in selected if row["backend"] == backend]
                returned = [row for row in attempts if scalar(row) is not None]
                used = [paired[repeat][backend] for repeat in matches] if matches else returned
                backends[backend] = {
                    "attempts": len(attempts),
                    "status_counts": dict(Counter(row["status"] for row in attempts)),
                    "distances": unique(scalar(row) for row in attempts),
                    "history_digests": unique(
                        row.get("outcome", {}).get("details", {}).get("histories_digest")
                        for row in attempts
                    ),
                    "operation_digests": unique(
                        row.get("outcome", {}).get("details", {}).get("operations_digest")
                        for row in attempts
                    ),
                    "metrics": {
                        metric: stats(row.get("measurement", {}).get(metric) for row in used)
                        for metric in METRICS
                    },
                    "excluded_repetitions": [
                        row["repeat"] for row in attempts if row["repeat"] not in matches
                    ],
                    "rss_reset_errors": unique(
                        row.get("measurement", {}).get("rss_peak_reset_error") for row in attempts
                    ),
                    "sampled_lifetime_peak_rss_bytes": stats(
                        row.get("sampled_lifetime_peak_rss_bytes") for row in attempts
                    ),
                }
                for row in returned:
                    distance = scalar(row)
                    rendered = format(distance, ".2f") if finite(distance) else None
                    historical_audit.append(
                        {
                            "case": case,
                            "collapse_corners": mode,
                            "backend": backend,
                            "repeat": row["repeat"],
                            "status": row["status"],
                            "fingerprint_capture": row.get("semantic_verification"),
                            "paired_semantics_verified": row["repeat"] in matches,
                            "observed_distance": distance,
                            "two_decimal_rendering": rendered,
                            "printed_distance": target["printed_distance"],
                            "difference_from_printed_numeric_value": distance
                            - float(target["printed_distance"])
                            if finite(distance)
                            else None,
                            "printed_format_agrees": rendered == target["printed_distance"]
                            if rendered is not None
                            else None,
                        }
                    )
            ref = backends["reference"]["metrics"]["align_wall_seconds"]["median"]
            rust = backends["native"]["metrics"]["align_wall_seconds"]["median"]
            memory_matches = [
                repeat
                for repeat in matches
                if all(
                    finite(row["measurement"].get("rss_peak_bytes"))
                    and row["measurement"]["rss_peak_bytes"] > 0
                    and row["measurement"].get("rss_peak_reset_error") is None
                    for row in paired[repeat].values()
                )
            ]
            memory_medians = {
                backend: statistics.median(
                    paired[repeat][backend]["measurement"]["rss_peak_bytes"]
                    for repeat in memory_matches
                )
                if memory_matches
                else None
                for backend in ("reference", "native")
            }
            mode_rows.append(
                {
                    "case": case,
                    "display_name": targets["models"][target["parent1"]]["display_name"]
                    + " × "
                    + targets["models"][target["parent2"]]["display_name"],
                    "collapse_corners": mode,
                    "printed_nodes": target["printed_node_counts"],
                    "backends": backends,
                    "verification": verification,
                    "matched_repetitions": matches,
                    "matched_count": len(matches),
                    "sample_scope": "matched_complete_returns"
                    if matches
                    else "unpaired_or_unverified_observed_returns",
                    "result_kind": "pilot"
                    if len(matches) == 1
                    else "repeated"
                    if matches
                    else "no_eligible_comparison",
                    "speedup": ref / rust if matches else None,
                    "memory_ratio": {
                        "metric": "absolute_alignment_window_peak_rss_bytes",
                        "matched_repetitions": memory_matches,
                        "matched_medians": memory_medians,
                        "python_over_rust": memory_medians["reference"] / memory_medians["native"]
                        if memory_matches
                        else None,
                        "note": "Ratio of matched medians, including baseline; no noisy incremental-RSS ratio.",
                    },
                }
            )
        modes["on" if mode else "off"] = mode_rows
    for off, on in zip(modes["off"], modes["on"], strict=True):
        for backend in ("reference", "native"):
            before, after = off["backends"][backend], on["backends"][backend]
            changes = {}
            for name in ("distances", "history_digests", "operation_digests"):
                changes[name] = (
                    before[name] != after[name] if before[name] and after[name] else None
                )
            delta = {}
            for metric in METRICS:
                left, right = (
                    before["metrics"][metric]["median"],
                    after["metrics"][metric]["median"],
                )
                delta[metric] = right - left if finite(left) and finite(right) else None
            sensitivity.append(
                {
                    "case": off["case"],
                    "backend": backend,
                    "off": before,
                    "on": after,
                    "changed": changes,
                    "on_minus_off_observed_medians": delta,
                    "note": "Outcome-sensitive observations, not an equivalent-algorithm speedup.",
                }
            )
    return {
        "schema": "rcswx-table42-summary-v1",
        "campaign_identity_sha256": report["campaign_identity_sha256"],
        "runtime_config": report["runtime_config"],
        "budget": report["budget"],
        "historical_targets": targets,
        "modes": modes,
        "sensitivity": sensitivity,
        "historical_audit": historical_audit,
        "historical_formatting_assumption": "Python fixed-point .2f on the observed binary float; this is not recovered historical formatting evidence.",
        "configurations": report["configurations"],
        "failures_and_exclusions": [
            row
            for row in rows
            if row["status"] != "complete"
            or row.get("outcome", {}).get("status") != "returned"
            or any(
                item["case"] == row["case"]
                and item["collapse_corners"] == row["collapse_corners"]
                and item["repeat"] == row["repeat"]
                and item["status"] == "mismatch"
                for item in verified
            )
        ],
        "comparisons": verified,
    }


def fmt(value, digits=3, divisor=1):
    return format(value / divisor, f".{digits}f") if finite(value) else "—"


def distances(backend):
    return (
        ", ".join(
            str(value) if finite(value) else json.dumps(value) for value in backend["distances"]
        )
        or "—"
    )


def metric(row, backend, name):
    return row["backends"][backend]["metrics"][name]["median"]


def status(row):
    if any(item["status"] == "mismatch" for item in row["verification"]):
        return "SEMANTIC MISMATCH; ratios suppressed"
    if row["matched_count"]:
        return f"full match; n={row['matched_count']}" + (
            " (pilot)" if row["matched_count"] == 1 else ""
        )
    return "; ".join(
        label + ": " + (", ".join(data["status_counts"]) or "not_run")
        for label, data in (
            ("Py", row["backends"]["reference"]),
            ("Rust", row["backends"]["native"]),
        )
    )


def time_cells(row):
    speedup = fmt(row["speedup"], 2) + ("×" if row["speedup"] is not None else "")
    historical = row.get("historical_time_comparison")
    if row["speedup"] is None and historical is not None:
        speedup = fmt(historical["ratio"], 2) + "×²"
    return [
        row["display_name"],
        distances(row["backends"]["reference"]),
        distances(row["backends"]["native"]),
        fmt(metric(row, "reference", "align_wall_seconds")),
        fmt(metric(row, "native", "align_wall_seconds")),
        fmt(metric(row, "reference", "rss_peak_bytes"), 1, 1024**2),
        fmt(metric(row, "native", "rss_peak_bytes"), 1, 1024**2),
        speedup,
        fmt(row["memory_ratio"]["python_over_rust"], 2)
        + ("×" if row["memory_ratio"]["python_over_rust"] is not None else ""),
        status(row),
    ]


def memory_cells(row):
    cells = [row["display_name"]]
    for backend in ("reference", "native"):
        cells.append(
            " / ".join(
                fmt(metric(row, backend, key), 1, 1024**2)
                for key in ("baseline_rss_bytes", "rss_peak_bytes")
            )
        )
    cells.extend(
        fmt(metric(row, backend, "rss_peak_delta_bytes"), 1, 1024**2)
        for backend in ("reference", "native")
    )
    cells.append(
        " / ".join(
            fmt(metric(row, backend, "retained_rss_bytes"), 1, 1024**2)
            for backend in ("reference", "native")
        )
    )
    ratio = row["memory_ratio"]
    cells.append(
        fmt(ratio["python_over_rust"], 2)
        + (
            f"× (n={len(ratio['matched_repetitions'])})"
            if ratio["python_over_rust"] is not None
            else ""
        )
    )
    return cells


def markdown_table(headers, rows):
    def line(values):
        return (
            "| "
            + " | ".join(str(value).replace("|", "\\|").replace("\n", " ") for value in values)
            + " |"
        )

    return "\n".join([line(headers), line(["---"] * len(headers)), *(line(row) for row in rows)])


TIME_HEADERS = [
    "Pair",
    "Python distance",
    "Rust distance",
    "Python median s",
    "Rust median s",
    "Python peak RSS MiB",
    "Rust peak RSS MiB",
    "Speedup",
    "Peak-RSS reduction",
    "Verification / status",
]
MEMORY_HEADERS = [
    "Pair",
    "Python base / peak MiB",
    "Rust base / peak MiB",
    "Python delta MiB",
    "Rust delta MiB",
    "Retained MiB: Py / Rust",
    "Peak ratio Py/Rust",
]


def render_svg(path, mode, rows, config):
    historical_notes = [
        row["historical_time_comparison"]["note"]
        for row in rows
        if row.get("historical_time_comparison") is not None and row["speedup"] is None
    ]
    height = 800 + 28 * len(historical_notes)
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="2290" height="{height}" viewBox="0 0 2290 {height}">',
        f'<rect width="2290" height="{height}" fill="white"/>',
        '<g fill="#111" font-family="DejaVu Serif,serif">',
    ]

    def text(x, y, value, size=18, weight="normal"):
        elements.append(
            f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}">{html.escape(str(value))}</text>'
        )

    def table(top, widths, headers, records):
        positions, offset = [], 24
        for width in widths:
            positions.append(offset)
            offset += width
        elements.append(
            f'<path d="M24 {top}H{offset} M24 {top + 47}H{offset} M24 {top + 247}H{offset}" stroke="#111" stroke-width="1.5"/>'
        )
        for x, header in zip(positions, headers, strict=True):
            text(x + 5, top + 31, header, 16, "bold")
        for index, record in enumerate(records):
            for x, value in zip(positions, record, strict=True):
                text(x + 5, top + 81 + index * 49, value, 17 if len(str(value)) < 33 else 13)

    text(24, 39, f"Table 4.2 reproduction — corner collapse {mode}", 28, "bold")
    text(
        24,
        71,
        f"Current repaired Python reference versus Rust · {config['profile_id']} · CPU {config['affinity_cpu']}",
        17,
    )
    table(
        97,
        [345, 170, 155, 165, 155, 210, 210, 165, 210, 457],
        TIME_HEADERS,
        [time_cells(row) for row in rows],
    )
    text(24, 387, "Alignment-window resident memory (MiB)", 23, "bold")
    table(
        410,
        [395, 325, 325, 245, 245, 420, 287],
        MEMORY_HEADERS,
        [memory_cells(row) for row in rows],
    )
    text(
        24,
        697,
        "Call timer ends at complete alignment-object return. Setup and fingerprinting are excluded; returned objects remain live.",
        16,
    )
    text(
        24,
        725,
        "Peak RSS is reset after setup and includes baseline. Reduction = Python/Rust matched median peak; censored or missing peaks are excluded.",
        16,
    )
    text(
        24,
        753,
        "Unmarked speedups and memory factors use matched completed repetitions. Unpaired times are descriptive; ² marks a historical/current ratio.",
        16,
    )
    text(
        24,
        781,
        "Historical algorithm, input configuration, pruning and timing boundary remain unknown; historical ratios are not controlled speedups.",
        16,
    )
    for index, note in enumerate(historical_notes):
        text(24, 809 + 28 * index, note, 16)
    elements.extend(["</g>", "</svg>"])
    path.write_text("\n".join(elements) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--targets", type=Path, default=ROOT / "tests/fixtures/table42/targets.json"
    )
    args = parser.parse_args()
    report, targets = json.loads(args.input.read_text()), json.loads(args.targets.read_text())
    if targets != verify_assets():
        raise ValueError("Report targets differ from the immutable supplied bundle")
    summary = summarize(report, targets)
    audits = [
        json.loads((args.input.parent / f"audit-{symbol}.json").read_text())
        for symbol in targets["models"]
    ]
    if any(audit["runtime_config_sha256"] != report["runtime_config_sha256"] for audit in audits):
        raise ValueError("Fixture audits do not match the measured runtime configuration")
    audit_rows = []
    for audit in audits:
        reference, native = audit["rows"]
        records = [row.get("fixture", {}) for row in (reference, native)]
        audit_rows.append(
            [
                targets["models"][audit["model"]]["display_name"],
                audit["model"],
                targets["models"][audit["model"]]["static_operation_occurrences"],
                records[0].get("runtime_node_count"),
                records[1].get("runtime_node_count"),
                " / ".join(str(row.get("alignment_tokens_including_start")) for row in records),
                audit["status"],
            ]
        )
    summary["fixture_audits"] = [
        {
            key: audit[key]
            for key in (
                "model",
                "status",
                "fixture_equal",
                "forward_equal",
                "runtime_config_sha256",
            )
        }
        for audit in audits
    ]
    destination = args.output_dir
    destination.mkdir(parents=True, exist_ok=True)
    dump(destination / "summary.json", summary)
    lines = [
        "# Table 4.2 — measured reproduction",
        "",
        "## Run identity",
        "",
        f"Host: `{report['host']}`. Source stratum: `{report['stratum']}`. Campaign: `{report['campaign_identity_sha256']}`.",
        f"Configuration: `{report['runtime_config']['profile_id']}`; CPU {report['affinity_cpu']}; one thread, seed 42, float32 CPU input `(2,3,32,32)`.",
        f"Native extension: `{report['native_extension']}`; SHA-256 `{report['native_sha256']}`.",
        f"Budget: {report['budget']['spent_seconds']:.3f} / {report['budget']['limit_seconds']:.3f} whole-worker seconds; remaining {report['budget']['remaining_seconds']:.3f} seconds.",
        f"Worker RSS guard: {report['rss_limit_mib'] / 1024:g} GiB. Peak RSS columns show recorded alignment-window high-water marks, including baseline. Missing window peaks remain unavailable; sampled lifetime peaks are separate diagnostics.",
        "",
        "Complete runtime/source/environment identities, raw repetitions and supervisor limits are in the input campaign JSON and summary JSON.",
        "",
        "## Historical targets — verbatim, not measured here",
        "",
        markdown_table(
            ["Pair", "Printed nodes", "Printed compute time", "Printed distance"],
            [
                [
                    row["pair_id"],
                    " / ".join(map(str, row["printed_node_counts"])),
                    row["printed_compute_time"],
                    row["printed_distance"],
                ]
                for row in targets["ordered_pairs"]
            ],
        ),
        "",
        "The final unit remains **741.57 minutes**, literally 44,494.2 seconds. Its unit is unconfirmed. Historical algorithm/configuration, corner flag, hardware, timing boundary and rounding convention are unknown. No historical/new timing ratio is a controlled speedup.",
        "",
        "The ordinary grammar rejects the supplied Mixers' relative linear factories. The separately named diagnostic profile admits exactly `linear_x(4)` and `linear_x(0.25)` in benchmark-local grammar copies; supplied source and reference algorithm bytes remain unchanged. Random fallback is forbidden.",
    ]
    provenance = report["runtime_config"].get("checkout_provenance")
    if provenance:
        lines += [
            "",
            "## Exact source provenance",
            "",
            f"Native base `{provenance['native_commit']}` plus tracked task diff `{provenance['tracked_task_diff_sha256']}`; Python reference `{provenance['reference_commit']}`.",
            "Exact source-file hashes and original worktree state are retained in the resolved configuration. The installed release extension is independently hashed.",
        ]
    lines += [
        "",
        "## Fixture audit",
        "",
        markdown_table(
            [
                "Model",
                "Source symbol",
                "Static count",
                "Python nodes",
                "Rust nodes",
                "Tokens Py / Rust",
                "Forward / replay",
            ],
            audit_rows,
        ),
        "",
        "Each model audit preserves source-to-factory-to-runtime mapping, exact topology and alias-sensitive metadata, complete fallback-free replay, independent backend ownership and exact tensor/state/RNG comparisons. Static/runtime node counts are not alignment token or model parameter counts. See the adjacent `audit-*.json` artifacts.",
    ]
    for mode, rows in summary["modes"].items():
        lines += [
            "",
            f"## Paired current reference — corner collapse {mode}",
            "",
            markdown_table(TIME_HEADERS, [time_cells(row) for row in rows]),
            "",
            markdown_table(MEMORY_HEADERS, [memory_cells(row) for row in rows]),
            "",
            f"![Observed table, corner collapse {mode}](table42-{mode}.svg)",
        ]
        render_svg(
            destination / f"table42-{mode}.svg",
            f"{mode}; {report['rss_limit_mib'] / 1024:g} GiB RSS guard",
            rows,
            report["runtime_config"],
        )
    lines += [
        "",
        "Call timing ends at complete alignment-object return, retaining ordered histories and edits/dependencies. Setup and fingerprinting are excluded. Whole-worker budget includes both. RSS peak is reset after setup, includes baseline, and excludes fingerprinting. Missing/reset-failed metrics are null; sampled lifetime peaks are separate diagnostics, never substituted for window peaks.",
        "",
        "Unmarked speedups and memory factors use only same-configuration, same-host, same-limit completed repetitions with matching distance, ordered histories, edits/dependencies, parent effects and RNG. One match is a pilot. Where no match exists, available returned-call timings are descriptive and may be unverified. Summary JSON includes all raw metric samples/min/median/max, exclusions, component verification and reset errors.",
        "",
        "Peak-RSS reduction is Python peak / Rust peak, using matched median alignment-window peaks including baseline. A factor above 1 means Rust uses less peak RSS; missing, reset-failed, censored or mismatched results have no factor.",
        "",
        "## Off/on sensitivity",
        "",
        "Changes are outcome-sensitive, not unconditional speedups. Full-precision distances, history/edit digest changes, status changes and observed median time/RSS deltas are in `summary.json` under `sensitivity`. The fixed `0.25` corner threshold is not a claimed percentage saving.",
        "",
        "## Historical distance audit",
        "",
        summary["historical_formatting_assumption"],
        "",
        "Every available full-precision distance, two-decimal rendering, printed-value difference and verification status is retained in `historical-audit.csv` and `summary.json`. Fingerprint capture records available evidence; only `paired_semantics_verified` confirms a matched comparison. Numerical agreement does not establish historical provenance.",
        "",
        "## Failures and exclusions",
        "",
        markdown_table(
            ["Pair", "Corner", "Backend", "Status", "Attempts", "Successful repetitions"],
            [
                [
                    row["case"],
                    "on" if row["collapse_corners"] else "off",
                    row["backend"],
                    row["status"],
                    row["attempts"],
                    row["successful_repetitions"],
                ]
                for row in report["configurations"]
            ],
        ),
        "",
        "Raw campaign rows retain error types, messages, stages, exit codes, limits, commands and per-worker log/sidecar paths. Matching exceptions are not successful distance results; fingerprinting timeouts are not labelled alignment timeouts.",
        "",
        "## Conclusions — separate claims",
        "",
        "Source recovery: immutable bundle checksums and supplied architecture definitions were verified; the historical algorithm and experimental configuration were not recovered.",
        "",
        f"Runtime fidelity: {sum(audit['fixture_equal'] and audit['forward_equal'] for audit in audits)}/{len(audits)} model audits passed exact replay/fixture and forward comparisons under the named diagnostic profile. This does not make the ordinary grammar accept the Mixers.",
        "",
        "Python/Rust equivalence: "
        + "; ".join(
            f"collapse {mode}: {sum(row['matched_count'] > 0 for row in rows)}/{len(rows)} pairs have fully verified matched completions"
            for mode, rows in summary["modes"].items()
        )
        + ". Censored or unpaired results do not establish equivalence.",
        "",
        "Historical reproduction: the distance audit records numerical agreement only. Neither matching printed decimals nor selecting a corner mode establishes historical provenance, timing-boundary equivalence or confirmation of the final printed time unit.",
        "",
        "Controlled performance: use only the eligible current-reference comparisons above. Sample counts are explicit; results do not establish general scaling laws or controlled historical speedups.",
        "",
        "## Reproduction artifacts",
        "",
        f"Raw observations: `{args.input.name}`. Persistent budget/reservations: `campaign-budget.json`. Exact build, validation and campaign commands: `*-invocation.json`. Resolve these names in the evidence directory containing the input JSON.",
        "",
        "Configuration and provenance: `runtime-resolved.json`, `runtime-strict.json`, `source-provenance.json`, `source-changes.patch`, `hardware.json`, `preparation.json`. Construction and validation: `strict-construction-probe.json`, `audit-*.json`, `final-tests.xml`, `report-tests.xml`, `report-validation.json`.",
        "",
        "Each campaign's `*.captures/` directory preserves worker commands, logs, progress and result sidecars. The final `artifact-manifest.json` inventories evidence hashes; `campaign-completion.json` records final scope and artifact locations. `source-snapshot.tar.gz` preserves the measured harness plus report/test sources.",
    ]
    (destination / "REPORT.md").write_text("\n".join(lines) + "\n")
    with (destination / "historical-audit.csv").open("w") as stream:
        fields = (
            list(summary["historical_audit"][0])
            if summary["historical_audit"]
            else ["case", "observed_distance", "printed_distance"]
        )
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary["historical_audit"])
    print(
        json.dumps(
            {
                "report": str(destination / "REPORT.md"),
                "matched_repetitions": {
                    mode: [row["matched_count"] for row in rows]
                    for mode, rows in summary["modes"].items()
                },
            }
        )
    )


if __name__ == "__main__":
    main()
