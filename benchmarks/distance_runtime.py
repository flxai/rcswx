"""Render the four-version frozen-corpus distance comparison and its analysis."""

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from scipy.stats import gaussian_kde

BACKGROUND = "#fcfaf3"
COLORS = {"original": "#365d9d", "v0.1": "#d78200", "v0.3": "#27816b", "v0.5": "#8952a0"}
LABELS = {
    "original": "Original RCSWX",
    "v0.1": "Rust v0.1 RCSWX (serial)",
    "v0.3": "Rust v0.3 RCSWX (serial)",
    "v0.5": "Rust v0.5 RCSWX (parallel, 10 workers)",
}
DOTTED_LEVELS = [1e1, 1e-1, 1e-2, 1e-3, 1e-4]


def positive(value):
    return isinstance(value, (int, float)) and math.isfinite(value) and value > 0


def analyze(report):
    series = {item["key"]: item for item in report["series"]}
    if set(series) != set(COLORS):
        raise ValueError("Expected original, v0.1, v0.3 and v0.5")
    indexed = {}
    for key, item in series.items():
        indexed[key] = {row["pair"]: row for row in item["rows"]}
        if len(indexed[key]) != len(item["rows"]):
            raise ValueError(f"Duplicate pair in {key}")
        expected_workers = 10 if key == "v0.5" else 1
        if item["workers"] != expected_workers:
            raise ValueError(f"Incorrect worker count for {key}")
    pairs = set(indexed["original"])
    if any(set(rows) != pairs for rows in indexed.values()):
        raise ValueError("All four series must cover the same frozen pair IDs")
    matched, comparable, mismatches = set(), set(), []
    for pair in sorted(pairs):
        rows = [indexed[key][pair] for key in COLORS]
        if len({(row["nodes"], row["pair_sha256"]) for row in rows}) != 1:
            raise ValueError(f"Input differs across versions: {pair}")
        if not all(row["status"] == "ok" for row in rows):
            continue
        comparable.add(pair)
        if len({row["distance"] for row in rows}) != 1:
            mismatches.append(
                {"pair": pair, "distances": {key: indexed[key][pair]["distance"] for key in COLORS}}
            )
            continue
        if all(not row.get("noop", False) and positive(row.get("wall_seconds")) for row in rows):
            matched.add(pair)
    summary = {
        "pairs": len(pairs),
        "comparable": len(comparable),
        "matched_nontrivial": len(matched),
        "mismatches": mismatches,
        "series": {},
    }
    for key, item in series.items():
        rows = item["rows"]
        ratios = [
            indexed["original"][pair]["wall_seconds"] / indexed[key][pair]["wall_seconds"]
            for pair in sorted(matched)
        ]
        reports = [row["native_execution"] for row in rows if row.get("native_execution")]
        summary["series"][key] = {
            "statuses": dict(Counter(row["status"] for row in rows)),
            "noop_completed": sum(row["status"] == "ok" and row.get("noop", False) for row in rows),
            "original_over_variant_median": float(np.median(ratios)) if ratios else None,
            "median_success_wall_seconds": float(
                np.median(
                    [
                        row["wall_seconds"]
                        for row in rows
                        if row["status"] == "ok" and not row.get("noop", False)
                    ]
                )
            ),
            "native_calls_with_parallel_cells": sum(item["parallel_cells"] > 0 for item in reports),
            "native_peak_jobs": max((item["peak_jobs"] for item in reports), default=0),
            "native_worker_limits": sorted({item["worker_limit"] for item in reports}),
        }
    return summary, matched


def distribution(ax, histogram):
    nodes = np.array(sorted(map(int, histogram)))
    counts = np.array([histogram[str(node)] for node in nodes])
    values = np.repeat(nodes, counts)
    ax.bar(
        nodes,
        counts / counts.sum(),
        width=1,
        color="#817493",
        alpha=0.32,
        linewidth=0,
        label="Population histogram",
    )
    if len(nodes) > 1 and len(values) > 1:
        kde = gaussian_kde(values)
        kde.set_bandwidth(bw_method=0.4 * kde.scotts_factor())
        grid = np.linspace(nodes.min(), nodes.max(), 800)
        ax.plot(grid, kde(grid), color="#817493", linewidth=1.5, label="KDE")
    ax.set_ylabel("Density")
    ax.set_xlabel("Nodes per parent (equal-sized pairs)")
    ax.yaxis.set_major_locator(MaxNLocator(3, prune="upper"))
    ax.xaxis.set_major_locator(MaxNLocator(9, integer=True))
    ax.set_xlim(max(0, nodes.min() - 3), nodes.max() + 3)
    ax.legend(loc="upper right", ncol=2, frameon=False, fontsize=10)
    ax.grid(axis="y", alpha=0.18)
    return int(counts.sum()), int(nodes.min()), int(nodes.max())


def render(report, destination, summary, matched):
    matplotlib.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#999999",
            "axes.facecolor": BACKGROUND,
            "figure.facecolor": BACKGROUND,
            "svg.fonttype": "none",
            "svg.hashsalt": "rcswx-distance-runtime-v1",
        }
    )
    fig, (ax, histogram) = plt.subplots(
        2, 1, figsize=(14, 9), sharex=True, gridspec_kw={"height_ratios": [3.5, 1], "hspace": 0.08}
    )
    ax.set_yscale("log")
    plotted = []
    for item in report["series"]:
        key, rows = item["key"], item["rows"]
        color = COLORS[key]
        complete = [
            row
            for row in rows
            if row["status"] == "ok"
            and not row.get("noop", False)
            and positive(row.get("wall_seconds"))
        ]
        noop = [
            row
            for row in rows
            if row["status"] == "ok"
            and row.get("noop", False)
            and positive(row.get("wall_seconds"))
        ]
        if complete:
            ax.scatter(
                [row["nodes"] for row in complete],
                [row["wall_seconds"] for row in complete],
                s=14,
                alpha=0.28,
                color=color,
                linewidths=0,
                zorder=3,
            )
            groups = defaultdict(list)
            for row in complete:
                if row["pair"] in matched:
                    groups[row["nodes"]].append(row["wall_seconds"])
            gx = sorted(groups)
            if gx:
                median = [np.median(groups[node]) for node in gx]
                q10 = [np.quantile(groups[node], 0.1) for node in gx]
                q90 = [np.quantile(groups[node], 0.9) for node in gx]
                ax.plot(gx, median, color=color, linewidth=1.6, marker="o", markersize=3, zorder=4)
                ax.fill_between(gx, q10, q90, color=color, alpha=0.09, linewidth=0, zorder=1)
            plotted.extend(row["wall_seconds"] for row in complete)
        if noop:
            ax.scatter(
                [row["nodes"] for row in noop],
                [row["wall_seconds"] for row in noop],
                s=22,
                facecolors="none",
                edgecolors=color,
                linewidths=0.8,
                alpha=0.6,
                zorder=3,
            )
            plotted.extend(row["wall_seconds"] for row in noop)
        for status, marker in [("timeout", "^"), ("rss_limit", "x"), ("exception", "s")]:
            points = [
                (
                    row["nodes"],
                    row.get("wall_seconds")
                    if status == "exception"
                    else row.get("observed_wall_lower_bound"),
                )
                for row in rows
                if row["status"] == status
            ]
            points = [(nodes, seconds) for nodes, seconds in points if positive(seconds)]
            if points:
                style = (
                    {"facecolors": "none", "edgecolors": color}
                    if status == "exception"
                    else {"color": color}
                )
                ax.scatter(
                    [point[0] for point in points],
                    [point[1] for point in points],
                    marker=marker,
                    s=26,
                    linewidths=0.8,
                    alpha=0.65,
                    zorder=5,
                    **style,
                )
                plotted.extend(point[1] for point in points)
    low = min(1e-5, 10 ** math.floor(math.log10(min(plotted) * 0.7))) if plotted else 1e-5
    high = max(100, 10 ** math.ceil(math.log10(max(plotted) * 1.3))) if plotted else 100
    ax.set_ylim(low, high)
    for value in DOTTED_LEVELS:
        ax.axhline(
            value,
            color="#888888",
            linewidth=0.9,
            linestyle=":",
            dash_capstyle="round",
            alpha=0.75,
            zorder=0,
        )
    for seconds, label in [(1, "1 sec"), (60, "1 min")]:
        ax.axhline(
            seconds, linestyle=(0, (5, 5)), color="#bd5750", alpha=0.55, linewidth=1, zorder=0
        )
        ax.text(
            1.006,
            seconds,
            label,
            transform=ax.get_yaxis_transform(),
            va="center",
            fontsize=10,
            color="#aa4d47",
        )
    ax.set_ylabel("Elapsed wall time (seconds, log scale)")
    handles = [
        Line2D(
            [], [], color=COLORS[key], marker="o", markersize=4, linewidth=1.7, label=LABELS[key]
        )
        for key in COLORS
    ]
    fig.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.073, 0.88),
        ncol=2,
        columnspacing=2,
        fontsize=10,
        frameon=False,
    )
    statuses = [
        Line2D(
            [],
            [],
            color="#666666",
            marker=marker,
            markerfacecolor="none" if marker in ("o", "s") else "#666666",
            linestyle="none",
            markersize=5,
            label=label,
        )
        for marker, label in [
            ("o", "No-op fast path"),
            ("^", "Wall limit (lower bound)"),
            ("x", "RSS limit (lower bound)"),
            ("s", "Raised exception"),
        ]
    ]
    fig.legend(
        handles=statuses,
        loc="upper left",
        bbox_to_anchor=(0.073, 0.807),
        ncol=4,
        fontsize=9,
        frameon=False,
    )
    counts_text = []
    for key in COLORS:
        item = summary["series"][key]
        counts = item["statuses"]
        limits = counts.get("timeout", 0) + counts.get("rss_limit", 0)
        other = sum(
            count
            for status, count in counts.items()
            if status not in {"ok", "exception", "timeout", "rss_limit"}
        )
        label = "Original" if key == "original" else f"Rust {key}"
        counts_text.append(
            f"{label}: {counts.get('ok', 0)} returned ({item['noop_completed']} no-ops), {counts.get('exception', 0)} exceptions, {limits} limited"
            + (f", {other} other failures" if other else "")
        )
    fig.text(
        0.937,
        0.877,
        "\n".join(counts_text),
        ha="right",
        va="top",
        fontsize=9,
        color="#444444",
        linespacing=1.35,
    )
    fig.suptitle(
        "RCSWX distance / alignment — original vs Rust-backed",
        x=0.08,
        y=0.97,
        ha="left",
        fontsize=18,
    )
    metadata = report["metadata"]
    subtitle = f"{metadata['host']} · serial: CPU {metadata['serial_cpu']} · parallel: 10 physical cores · {metadata['timeout_wall_seconds']:g} s wall / {metadata['memory_limit_bytes'] / 1024**3:g} GiB RSS · ref {metadata['reference']['source_commit'][:7]}"
    fig.text(0.08, 0.919, subtitle, fontsize=10, color="#555555")
    population, minimum, maximum = distribution(histogram, metadata["population_histogram"])
    ratios = "; ".join(
        f"{key}: {summary['series'][key]['original_over_variant_median']:.2f}×"
        for key in ("v0.1", "v0.3", "v0.5")
        if summary["series"][key]["original_over_variant_median"] is not None
    )
    caption = (
        f"{population:,} population entries · {minimum}–{maximum} nodes · {summary['pairs']:,} seeded pairs per version; {summary['comparable']:,} jointly returned, {len(summary['mismatches'])} distance disagreements.\n"
        f"Matched nontrivial median original/version wall-time ratios: {ratios} (n={len(matched)}).\n"
        "Lines: common successful, equal-distance, nontrivial per-node medians; bands: 10–90% observed ranges, not confidence intervals. No extrapolation.\n"
        "Fresh bounded child per pair; native pool startup included. Ten workers is a ceiling: ineligible waves fall back to serial. No network build/evaluation."
    )
    fig.text(0.08, 0.025, caption, fontsize=9, color="#444444", va="bottom", linespacing=1.3)
    fig.subplots_adjust(left=0.08, right=0.935, bottom=0.175, top=0.75)
    destination.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("svg", "png"):
        fig.savefig(
            destination.with_suffix("." + suffix),
            dpi=150,
            facecolor=BACKGROUND,
            metadata={"Date": None} if suffix == "svg" else None,
        )
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("distance-runtime"))
    args = parser.parse_args()
    report = json.loads(args.input.read_text())
    summary, matched = analyze(report)
    render(report, args.output, summary, matched)
    summary.update(
        schema="rcswx-distance-runtime-plot-v1",
        metric="wall_seconds",
        input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
        renderer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        matplotlib_version=matplotlib.__version__,
        dotted_grey_levels=DOTTED_LEVELS,
        files={suffix: str(args.output.with_suffix("." + suffix)) for suffix in ("svg", "png")},
    )
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
