from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from datasets import Dataset, DatasetDict, load_from_disk


BUCKET_ORDER = [0.25, 0.5, 0.75, 1.0]
CHANGE_ORDER = ["improved", "same", "regressed"]
CHANGE_COLORS = {
    "improved": "#2e8b57",
    "same": "#6b7280",
    "regressed": "#c2410c",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--reference_field", type=str, default="baseline_passrate")
    parser.add_argument("--conditioned_field", type=str, default="conditioned_passrate")
    return parser


def load_rows(path_str: str) -> list[dict]:
    path = Path(path_str)
    dataset = load_from_disk(str(path))
    if isinstance(dataset, DatasetDict):
        if "train" in dataset:
            dataset = dataset["train"]
        else:
            dataset = next(iter(dataset.values()))
    assert isinstance(dataset, Dataset)
    return [dict(row) for row in dataset]


def compute_bucket_stats(rows: list[dict], reference_field: str, conditioned_field: str) -> dict:
    bucket_stats = defaultdict(
        lambda: {
            "count": 0,
            "improved": 0,
            "same": 0,
            "regressed": 0,
            "mean_reference_sum": 0.0,
            "mean_conditioned_sum": 0.0,
            "delta_sum": 0.0,
            "transitions": Counter(),
            "conditioned_dist": Counter(),
        }
    )

    for row in rows:
        reference = float(row.get(reference_field, row.get("passrate", 0.0)))
        conditioned = float(row.get(conditioned_field, 0.0))
        stats = bucket_stats[reference]
        stats["count"] += 1
        stats["mean_reference_sum"] += reference
        stats["mean_conditioned_sum"] += conditioned
        stats["delta_sum"] += conditioned - reference
        if conditioned > reference:
            stats["improved"] += 1
        elif conditioned < reference:
            stats["regressed"] += 1
        else:
            stats["same"] += 1
        stats["transitions"][conditioned] += 1
        stats["conditioned_dist"][conditioned] += 1

    summary = {}
    for bucket in BUCKET_ORDER:
        stats = bucket_stats[bucket]
        count = stats["count"]
        summary[str(bucket)] = {
            "count": count,
            "mean_reference_passrate": stats["mean_reference_sum"] / count if count else 0.0,
            "mean_conditioned_passrate": stats["mean_conditioned_sum"] / count if count else 0.0,
            "mean_delta": stats["delta_sum"] / count if count else 0.0,
            "improved_count": stats["improved"],
            "same_count": stats["same"],
            "regressed_count": stats["regressed"],
            "improved_pct": stats["improved"] / count if count else 0.0,
            "same_pct": stats["same"] / count if count else 0.0,
            "regressed_pct": stats["regressed"] / count if count else 0.0,
            "conditioned_distribution": {
                str(value): stats["conditioned_dist"].get(value, 0) for value in BUCKET_ORDER + [0.0]
            },
        }
    return summary


def plot_change_breakdown(stats: dict, output_path: Path) -> None:
    labels = [str(bucket) for bucket in BUCKET_ORDER]
    x = np.arange(len(labels))
    width = 0.65
    bottoms = np.zeros(len(labels))

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for change in CHANGE_ORDER:
        values = np.array([stats[label][f"{change}_pct"] for label in labels]) * 100
        ax.bar(
            x,
            values,
            width=width,
            bottom=bottoms,
            color=CHANGE_COLORS[change],
            label=change.title(),
        )
        for idx, value in enumerate(values):
            if value >= 8:
                ax.text(
                    x[idx],
                    bottoms[idx] + value / 2,
                    f"{value:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="white",
                    fontweight="bold",
                )
        bottoms += values

    ax.set_title("Passrate Change Breakdown by Original Bucket")
    ax.set_xlabel("Original Baseline Passrate Bucket")
    ax.set_ylabel("Percent of Rows")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 100)
    ax.legend(frameon=False)
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_mean_before_after(stats: dict, output_path: Path) -> None:
    labels = [str(bucket) for bucket in BUCKET_ORDER]
    x = np.arange(len(labels))
    width = 0.34
    reference = [stats[label]["mean_reference_passrate"] for label in labels]
    conditioned = [stats[label]["mean_conditioned_passrate"] for label in labels]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ref_bars = ax.bar(x - width / 2, reference, width=width, color="#94a3b8", label="Original")
    cond_bars = ax.bar(x + width / 2, conditioned, width=width, color="#2563eb", label="Conditioned")

    for bars in (ref_bars, cond_bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.015,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set_title("Mean Passrate by Original Bucket")
    ax.set_xlabel("Original Baseline Passrate Bucket")
    ax.set_ylabel("Mean Passrate")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.08)
    ax.legend(frameon=False)
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_transition_heatmap(stats: dict, output_path: Path) -> None:
    rows = BUCKET_ORDER
    cols = [0.0, 0.25, 0.5, 0.75, 1.0]
    matrix = np.array(
        [
            [stats[str(row)]["conditioned_distribution"].get(str(col), 0) for col in cols]
            for row in rows
        ]
    )
    row_totals = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, row_totals, where=row_totals != 0) * 100

    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    im = ax.imshow(normalized, cmap="Blues", aspect="auto", vmin=0, vmax=max(1, normalized.max()))
    ax.set_title("Conditioned Passrate Distribution Within Each Original Bucket")
    ax.set_xlabel("Conditioned Passrate")
    ax.set_ylabel("Original Baseline Passrate Bucket")
    ax.set_xticks(np.arange(len(cols)), [str(col) for col in cols])
    ax.set_yticks(np.arange(len(rows)), [str(row) for row in rows])

    for i in range(len(rows)):
        for j in range(len(cols)):
            pct = normalized[i, j]
            count = matrix[i, j]
            if count:
                ax.text(
                    j,
                    i,
                    f"{count}\n{pct:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if pct > normalized.max() * 0.45 else "black",
                    fontweight="bold" if pct > 20 else None,
                )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Percent Within Original Bucket")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def write_summary_md(stats: dict, output_path: Path) -> None:
    lines = [
        "# Bucket Passrate Change Plots",
        "",
        "- `bucket_change_breakdown.png`: improved / same / regressed share within each original bucket",
        "- `bucket_mean_before_after.png`: mean original vs conditioned passrate by bucket",
        "- `bucket_transition_heatmap.png`: conditioned passrate distribution within each original bucket",
        "",
        "## Bucket Summary",
        "",
    ]
    for bucket in [str(value) for value in BUCKET_ORDER]:
        item = stats[bucket]
        lines.extend(
            [
                f"### Bucket {bucket}",
                f"- count: `{item['count']}`",
                f"- mean: `{item['mean_reference_passrate']:.3f} -> {item['mean_conditioned_passrate']:.3f}`",
                f"- delta: `{item['mean_delta']:+.3f}`",
                f"- improved / same / regressed: `{item['improved_count']}` / `{item['same_count']}` / `{item['regressed_count']}`",
                "",
            ]
        )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(args.input_path)
    stats = compute_bucket_stats(rows, args.reference_field, args.conditioned_field)

    (output_dir / "bucket_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    plot_change_breakdown(stats, output_dir / "bucket_change_breakdown.png")
    plot_mean_before_after(stats, output_dir / "bucket_mean_before_after.png")
    plot_transition_heatmap(stats, output_dir / "bucket_transition_heatmap.png")
    write_summary_md(stats, output_dir / "README.md")

    print(json.dumps({"output_dir": str(output_dir), "num_rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
