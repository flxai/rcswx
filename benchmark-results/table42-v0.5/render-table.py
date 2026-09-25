"""Render published Table 4.2 times beside three measured, exact Rust releases."""

import csv
import hashlib
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox

ROOT = Path(__file__).resolve().parent
VERSIONS = ("v0.1", "v0.3", "v0.5")


def main():
    evidence_path = ROOT / "observations.json"
    evidence = json.loads(evidence_path.read_text())
    spec = json.loads((ROOT / "campaign.json").read_text())
    targets_path = ROOT.parents[1] / "tests/fixtures/table42/targets.json"
    targets = json.loads(targets_path.read_text())
    assert hashlib.sha256(targets_path.read_bytes()).hexdigest() == evidence["targets_sha256"]
    assert (
        hashlib.sha256((ROOT / "campaign.json").read_bytes()).hexdigest()
        == evidence["campaign_sha256"]
    )
    records, cells, summaries = [], [], []
    for target in targets["ordered_pairs"]:
        case = target["pair_id"]
        rows = [row for row in evidence["rows"] if row["case"] == case]
        assert len(rows) == 9
        assert all(
            row["status"] == "complete" and row["outcome"]["status"] == "returned" for row in rows
        )
        assert len({row["outcome"]["digest"] for row in rows}) == 1, f"Semantic mismatch: {case}"
        assert len({row["fixture_runtime_sha256"] for row in rows}) == 1, f"Input mismatch: {case}"
        assert all(row["nodes"] == target["printed_node_counts"] for row in rows)
        distance = rows[0]["outcome"]["details"]["distance"]
        assert all(row["outcome"]["details"]["distance"] == distance for row in rows)
        assert f"{distance:.2f}" == target["printed_distance"]
        name = ", ".join(
            targets["models"][target[key]]["display_name"] for key in ("parent1", "parent2")
        )
        record = {
            "case": case,
            "architectures": name,
            "nodes": ", ".join(map(str, target["printed_node_counts"])),
            "original_published_time": target["printed_compute_time"],
            "distance": distance,
        }
        summary = {
            "case": case,
            "distance": distance,
            "outcome_digest": rows[0]["outcome"]["digest"],
            "versions": {},
        }
        times = []
        for version in VERSIONS:
            selected = [row for row in rows if row["release"] == version]
            assert {row["repeat"] for row in selected} == {0, 1, 2}
            assert all(row["affinity"] == spec["releases"][version]["cpus"] for row in selected)
            seconds = [row["measurement"]["align_wall_seconds"] for row in selected]
            median = statistics.median(seconds)
            record[f"{version}_wall_seconds_median"] = median
            summary["versions"][version] = {
                "seconds": seconds,
                "median_seconds": median,
                "min_seconds": min(seconds),
                "max_seconds": max(seconds),
            }
            if version == "v0.5":
                reports = [row["measurement"]["native_execution"] for row in selected]
                assert all(
                    e["requested_workers"] == e["worker_limit"] == e["pool_capacity"] == 10
                    for e in reports
                )
                assert all(e["peak_jobs"] <= 10 for e in reports)
                summary["versions"][version]["execution"] = reports
            times.append(f"{median:.3f} s")
        published = target["printed_compute_time"].replace("seconds", "s").replace("minutes", "min")
        cells.append([name, record["nodes"], published, *times, f"{distance:.3f}"])
        records.append(record)
        summaries.append(summary)
    with (ROOT / "table42.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    plt.rcParams.update({"font.family": "DejaVu Serif", "svg.fonttype": "none", "font.size": 11})
    fig, ax = plt.subplots(figsize=(16, 5.2), facecolor="white")
    fig.subplots_adjust(left=0.025, right=0.975, bottom=0.02, top=0.98)
    ax.set_axis_off()
    fig.text(
        0.5,
        0.92,
        "Table 4.2: RCSWX on the selected predesigned architectures",
        ha="center",
        fontsize=17,
    )
    fig.text(0.628, 0.82, "Compute time", ha="center", fontsize=12)
    table = ax.table(
        cellText=cells,
        colLabels=[
            "Architectures",
            "Number\nof nodes",
            "Original\n(published)",
            "Rust v0.1\nserial",
            "Rust v0.3\nserial",
            "Rust v0.5\n10 workers",
            "Distance",
        ],
        colWidths=[0.27, 0.105, 0.135, 0.125, 0.125, 0.135, 0.105],
        cellLoc="center",
        bbox=Bbox.from_bounds(0, 0.36, 1, 0.43),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    for (row, column), cell in table.get_celld().items():
        cell.set_facecolor("white")
        cell.set_edgecolor("#777777")
        cell.set_linewidth(0.65)
        cell.visible_edges = "BT" if row == 0 else ("B" if row == len(cells) else "")
        if row == 0:
            cell.set_text_props(weight="bold")
        elif column == 0:
            cell.set_text_props(ha="left")
    fig.text(
        0.025,
        0.29,
        "Rust: median wall time of 3 fresh-process calls per cell; corner collapse ON; cold pool startup included; 16 GiB RSS guard.",
        fontsize=10,
    )
    fig.text(
        0.025,
        0.24,
        "ono · serial: E-core CPU 24 · parallel: 10 E-cores · Rust 1.98.1 · Python 3.12.14 · seed 42 · float32 input (2, 3, 32, 32)",
        fontsize=10,
    )
    fig.text(
        0.025,
        0.19,
        "Timing ends at Alignment return; input construction and later history traversal are excluded. All native outcomes match exactly.",
        fontsize=10,
    )
    fig.text(
        0.025,
        0.14,
        "Original values are transcribed from the supplied table, not remeasured. Historical hardware and timing boundary are unknown.",
        fontsize=10,
    )
    fig.text(
        0.025,
        0.09,
        "741.57 minutes is preserved literally; a possible unit typo remains unverified. Native distances round to the published distances.",
        fontsize=10,
    )
    fig.savefig(ROOT / "table42.svg", facecolor="white", metadata={"Date": None})
    fig.savefig(ROOT / "table42.png", dpi=180, facecolor="white")
    plt.close(fig)
    summary = {
        "schema": "rcswx-table42-four-columns-summary-v1",
        "observations_sha256": hashlib.sha256(evidence_path.read_bytes()).hexdigest(),
        "renderer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "original_column": "historical published values, not matched measurements",
        "native_metric": "median Alignment-return wall seconds, three repetitions",
        "cases": summaries,
    }
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
