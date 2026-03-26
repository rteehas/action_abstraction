from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean

from datasets import Dataset, DatasetDict, load_from_disk


def load_split_counts(dataset_path: Path) -> dict:
    data = load_from_disk(str(dataset_path))
    if isinstance(data, DatasetDict):
        splits = {name: ds for name, ds in data.items()}
    elif isinstance(data, Dataset):
        splits = {"train": data}
    else:
        raise TypeError(f"Unsupported dataset type: {type(data)}")

    summary = {"splits": {}, "selected_rows": 0}
    for split_name, ds in splits.items():
        rows = [dict(ds[i]) for i in range(len(ds))]
        counter = Counter(float(r.get("baseline_passrate", 0.0)) for r in rows)
        summary["splits"][split_name] = {
            "rows": len(rows),
            "baseline_bucket_counts": {str(k): counter[k] for k in sorted(counter)},
        }
        summary["selected_rows"] += len(rows)
    return summary


def load_final_checkpoint_metrics(run_root: Path, checkpoints: list[dict]) -> dict:
    if not checkpoints:
        return {}
    last = max(checkpoints, key=lambda row: row["step"])
    ckpt_dir = run_root / last["checkpoint"]
    state_path = ckpt_dir / "trainer_state.json"
    if not state_path.exists():
        return {"last_checkpoint": last["checkpoint"]}
    state = json.loads(state_path.read_text())
    log_history = state.get("log_history", [])
    eval_rows = [row for row in log_history if "eval_loss" in row]
    train_rows = [row for row in log_history if "loss" in row]
    best_eval = min(eval_rows, key=lambda row: row["eval_loss"]) if eval_rows else None
    final_eval = eval_rows[-1] if eval_rows else None
    final_train = train_rows[-1] if train_rows else None
    return {
        "last_checkpoint": last["checkpoint"],
        "best_eval": best_eval,
        "final_eval": final_eval,
        "final_train": final_train,
        "global_step": state.get("global_step"),
    }


def write_plots(output_dir: Path, checkpoints: list[dict]) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return []

    if not checkpoints:
        return []

    plot_paths = []
    epochs = [row["epoch"] for row in checkpoints]
    avg = [row["best_average_passrate"] for row in checkpoints]
    best4 = [row["best_best_of_4_passrate"] for row in checkpoints]
    aime = [row.get("aime_mean_at_best_average_temp") for row in checkpoints]
    amc = [row.get("amc_mean_at_best_average_temp") for row in checkpoints]
    eval_loss = [row.get("trainer_eval_loss") for row in checkpoints]
    eval_acc = [row.get("trainer_eval_mean_token_accuracy") for row in checkpoints]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, avg, marker="o", label="best average RLAD passrate")
    plt.plot(epochs, best4, marker="o", label="best-of-4 RLAD passrate")
    plt.xlabel("Epoch")
    plt.ylabel("Passrate")
    plt.title("RLAD passrate by checkpoint")
    plt.grid(alpha=0.3)
    plt.legend()
    path = output_dir / "checkpoint_rlad_passrate_curves.png"
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    plot_paths.append(str(path))

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, aime, marker="o", label="AIME mean at best avg temp")
    plt.plot(epochs, amc, marker="o", label="AMC mean at best avg temp")
    plt.xlabel("Epoch")
    plt.ylabel("Passrate")
    plt.title("AIME vs AMC by checkpoint")
    plt.grid(alpha=0.3)
    plt.legend()
    path = output_dir / "checkpoint_aime_amc_curves.png"
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    plot_paths.append(str(path))

    plt.figure(figsize=(8, 5))
    ax1 = plt.gca()
    ax1.plot(epochs, eval_loss, marker="o", color="tab:red", label="trainer eval loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Eval loss", color="tab:red")
    ax1.tick_params(axis="y", labelcolor="tab:red")
    ax1.grid(alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(epochs, eval_acc, marker="o", color="tab:blue", label="trainer eval token acc")
    ax2.set_ylabel("Eval token acc", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")
    plt.title("Trainer eval metrics by checkpoint")
    path = output_dir / "checkpoint_trainer_eval_curves.png"
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    plot_paths.append(str(path))

    return plot_paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--run_root", required=True)
    parser.add_argument("--family_summary", required=True)
    parser.add_argument("--baseline_summary", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--sweeps_root", required=True)
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    run_root = Path(args.run_root)
    family_summary_path = Path(args.family_summary)
    baseline_summary_path = Path(args.baseline_summary)
    output_dir = Path(args.output_dir)
    sweeps_root = Path(args.sweeps_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    family = json.loads(family_summary_path.read_text())
    baseline = json.loads(baseline_summary_path.read_text())
    checkpoints = family.get("checkpoints", [])
    dataset_summary = load_split_counts(dataset_path)
    training_metrics = load_final_checkpoint_metrics(run_root, checkpoints)
    plot_paths = write_plots(output_dir, checkpoints)

    best_avg = family.get("best_checkpoint_by_average")
    best_best = family.get("best_checkpoint_by_best_of_4")
    baseline_mean = baseline.get("baseline_mean_passrate")
    best_avg_delta = None if best_avg is None else best_avg["best_average_passrate"] - baseline_mean
    best_best_delta = None if best_best is None else best_best["best_best_of_4_passrate"] - baseline_mean

    payload = {
        "dataset_path": str(dataset_path),
        "run_root": str(run_root),
        "sweeps_root": str(sweeps_root),
        "dataset_summary": dataset_summary,
        "baseline_mean_passrate": baseline_mean,
        "num_checkpoints_evaluated": family.get("num_checkpoints_evaluated", 0),
        "best_checkpoint_by_average": best_avg,
        "best_checkpoint_by_best_of_4": best_best,
        "best_average_delta_vs_baseline": best_avg_delta,
        "best_best_of_4_delta_vs_baseline": best_best_delta,
        "training_metrics": training_metrics,
        "plots": plot_paths,
        "family_summary_path": str(family_summary_path),
        "baseline_summary_path": str(baseline_summary_path),
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Worklog 2026-03-22",
        "",
        "## Scope",
        "",
        "This note summarizes the non-regressed 9K-subset SFT run and the RLAD checkpoint sweep that followed it.",
        "",
        "## 1. SFT Dataset And Training Setup",
        "",
        f"Source conditioned-solving dataset:",
        f"- `{dataset_path}`",
        "",
        "Selection rule:",
        "- keep all rows where abstraction-conditioned passrate was greater than or equal to baseline passrate",
        "",
        f"Selected rows: `{dataset_summary['selected_rows']}`",
    ]
    for split_name, split_summary in dataset_summary["splits"].items():
        lines.extend([
            f"",
            f"{split_name.title()} split:",
            f"- rows: `{split_summary['rows']}`",
            f"- baseline bucket counts: `{split_summary['baseline_bucket_counts']}`",
        ])
    lines.extend([
        "",
        "SFT configuration:",
        "- base model: `Qwen/Qwen3-1.7B`",
        "- prompt template: `prompt_templates/sft_principle_generation.txt`",
        "- epochs: `5`",
        "- learning rate: `5e-5`",
        "- seed: `100`",
        "- save/eval cadence: every half epoch (`217` steps)`",
        "- expected checkpoints evaluated: `10`",
        f"- run root: `{run_root}`",
        "",
        "## 2. RLAD Evaluation Setup",
        "",
        "Evaluation slice:",
        "- dataset: `/workspace/rlad_aime_amc_scored`",
        "- sample: `12` problems, balanced `6` AIME + `6` AMC, fixed seed `100`",
        "- baseline solver: `Qwen/Qwen3-1.7B`, `4` samples, `temperature=0.6`, `top_p=0.95`, `top_k=20`, `max_tokens=32768`, thinking on",
        "- abstraction generator sweep: `4` abstractions per problem at temperatures `0.0`, `0.3`, `0.6`, `0.9`",
        "- conditioned solver: same solver config as baseline, thinking on",
        f"- sweeps root: `{sweeps_root}`",
        "",
        "## 3. Baseline",
        "",
        f"- baseline mean passrate on the 12-problem sample: `{baseline_mean:.4f}`",
    ])
    if best_avg is not None:
        lines.extend([
            "",
            "## 4. Best Checkpoints",
            "",
            f"Best checkpoint by average conditioned passrate:",
            f"- checkpoint: `{best_avg['checkpoint']}`",
            f"- epoch: `{best_avg['epoch']:.2f}`",
            f"- best average temp: `{best_avg['best_average_temperature']}`",
            f"- average RLAD passrate: `{best_avg['best_average_passrate']:.4f}`",
            f"- delta vs baseline: `{best_avg_delta:+.4f}`",
            f"- AIME mean at best avg temp: `{best_avg['aime_mean_at_best_average_temp']:.4f}`",
            f"- AMC mean at best avg temp: `{best_avg['amc_mean_at_best_average_temp']:.4f}`",
            "",
            f"Best checkpoint by best-of-4 abstraction passrate:",
            f"- checkpoint: `{best_best['checkpoint']}`",
            f"- epoch: `{best_best['epoch']:.2f}`",
            f"- best-of-4 temp: `{best_best['best_best_of_4_temperature']}`",
            f"- best-of-4 RLAD passrate: `{best_best['best_best_of_4_passrate']:.4f}`",
            f"- delta vs baseline: `{best_best_delta:+.4f}`",
            "",
            "## 5. Training Metrics",
            "",
        ])
        best_eval = training_metrics.get("best_eval")
        final_eval = training_metrics.get("final_eval")
        final_train = training_metrics.get("final_train")
        if best_eval is not None:
            lines.append(f"- best trainer eval loss: `{best_eval['eval_loss']:.4f}` at step `{best_eval['step']}`")
            if best_eval.get('eval_mean_token_accuracy') is not None:
                lines.append(f"- trainer eval token accuracy at best eval loss step: `{best_eval['eval_mean_token_accuracy']:.4f}`")
        if final_eval is not None:
            lines.append(f"- final trainer eval loss: `{final_eval['eval_loss']:.4f}` at step `{final_eval['step']}`")
            if final_eval.get('eval_mean_token_accuracy') is not None:
                lines.append(f"- final trainer eval token accuracy: `{final_eval['eval_mean_token_accuracy']:.4f}`")
        if final_train is not None:
            lines.append(f"- final logged train loss: `{final_train['loss']:.4f}`")
        lines.extend([
            "",
            "## 6. Checkpoint Table",
            "",
            "| checkpoint | epoch | eval loss | eval token acc | best avg temp | best avg | delta vs baseline | AIME | AMC | best-of-4 temp | best-of-4 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ])
        for row in checkpoints:
            delta = row['best_average_passrate'] - baseline_mean
            def f(x):
                return '' if x is None else f'{x:.4f}'
            lines.append(
                f"| {row['checkpoint']} | {row['epoch']:.2f} | {f(row.get('trainer_eval_loss'))} | {f(row.get('trainer_eval_mean_token_accuracy'))} | {row['best_average_temperature']} | {row['best_average_passrate']:.4f} | {delta:+.4f} | {f(row.get('aime_mean_at_best_average_temp'))} | {f(row.get('amc_mean_at_best_average_temp'))} | {row['best_best_of_4_temperature']} | {row['best_best_of_4_passrate']:.4f} |"
            )
        lines.extend([
            "",
            "## 7. Artifact Map",
            "",
            f"- family summary: `{family_summary_path}`",
            f"- checkpoint-family report: `{Path(family_summary_path).with_name('REPORT.md')}`",
            f"- plots: `{plot_paths}`",
            f"- per-checkpoint sweep roots: `{sweeps_root}`",
            "",
            "Per-checkpoint docs live under each `checkpoint_XXXX_sweep/final_summary/` directory:",
            "- `REPORT_BREAKDOWN.md`",
            "- `BEST_WORST_ABSTRACTIONS.md`",
            "- `summary.json`",
            "- `summary_enriched.json`",
        ])

    report_text = "\n".join(lines)
    (output_dir / "WORKLOG_20260322.md").write_text(report_text, encoding="utf-8")
    (output_dir / "REPORT.md").write_text(report_text, encoding="utf-8")
    print(json.dumps({
        "summary": str(output_dir / "summary.json"),
        "report": str(output_dir / "REPORT.md"),
        "worklog": str(output_dir / "WORKLOG_20260322.md"),
        "plots": plot_paths,
    }, indent=2))


if __name__ == "__main__":
    main()
