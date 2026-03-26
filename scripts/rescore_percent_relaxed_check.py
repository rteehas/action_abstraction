from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

spec = importlib.util.spec_from_file_location(
    "patched_reward",
    REPO_ROOT / "verl/verl/utils/reward_score/deepscaler_math_reward_multibox_patched.py",
)
patched_reward = importlib.util.module_from_spec(spec)
spec.loader.exec_module(patched_reward)

from process_deepscaler_dataset import answers_equivalent

PERCENT_WORD_RE = re.compile(r"\bpercent(?:age)?s?\b", flags=re.IGNORECASE)


def is_percent_like(*texts: str) -> bool:
    for text in texts:
        if not text:
            continue
        if "%" in text or r"\%" in text or PERCENT_WORD_RE.search(text):
            return True
    return False


def strip_percent_markers(text: str) -> str:
    text = text.strip()
    text = text.replace(r"\%", "")
    text = text.replace("%", "")
    text = PERCENT_WORD_RE.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def compute_score_percent_relaxed(solution_str: str, ground_truth: str, percent_tol: float = 0.005) -> float:
    base = patched_reward.compute_score(None, solution_str, ground_truth)
    if base == 1.0:
        return 1.0

    boxed_answers = patched_reward.extract_boxed_answers(solution_str)
    if not boxed_answers:
        return 0.0

    final_boxed = boxed_answers[-1]
    pred_raw = patched_reward.unbox(final_boxed)
    gt_raw = ground_truth.strip()

    if not is_percent_like(pred_raw, gt_raw):
        return 0.0

    stripped_pred = strip_percent_markers(pred_raw)
    stripped_gt = strip_percent_markers(gt_raw)
    if not stripped_pred or not stripped_gt:
        return 0.0

    try:
        if answers_equivalent(stripped_pred, stripped_gt, tol=percent_tol):
            return 1.0
    except Exception:
        return 0.0
    return 0.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--percent_tol", type=float, default=0.005)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = json.loads(Path(args.rows_path).read_text(encoding="utf-8"))

    changed_sample_count = 0
    changed_row_count = 0
    changed_regression_status_count = 0
    percent_like_sample_count = 0
    percent_like_changed_sample_count = 0
    flipped_examples = []
    bucket_summary: dict[str, dict[str, int]] = {}

    for row in rows:
        old_corrects = []
        new_corrects = []
        row_changed = False
        row_id = row.get("row_id")
        answer = row.get("answer", "")
        solver_samples = row.get("solver_samples") or []

        for sample in solver_samples:
            solution = sample.get("output_text") or ""
            predicted = sample.get("predicted_answer") or ""
            old = 1.0 if sample.get("correct") else 0.0
            new = compute_score_percent_relaxed(solution, answer, percent_tol=args.percent_tol)
            old_corrects.append(old)
            new_corrects.append(new)

            percent_like = is_percent_like(predicted, answer)
            if percent_like:
                percent_like_sample_count += 1

            if old != new:
                changed_sample_count += 1
                row_changed = True
                if percent_like:
                    percent_like_changed_sample_count += 1
                if len(flipped_examples) < 100:
                    flipped_examples.append(
                        {
                            "row_id": row_id,
                            "seed": sample.get("seed"),
                            "ground_truth": answer,
                            "predicted_answer": predicted,
                            "old_score": old,
                            "new_score": new,
                            "passrate": row.get("passrate"),
                            "old_mean_solver_accuracy": row.get("mean_solver_accuracy"),
                        }
                    )

        old_mean = sum(old_corrects) / len(old_corrects) if old_corrects else 0.0
        new_mean = sum(new_corrects) / len(new_corrects) if new_corrects else 0.0
        row["percent_relaxed_mean_solver_accuracy"] = new_mean
        row["percent_relaxed_num_correct"] = int(sum(new_corrects))

        if row_changed:
            changed_row_count += 1

        old_regressed = old_mean < float(row.get("passrate", 0.0))
        new_regressed = new_mean < float(row.get("passrate", 0.0))
        if old_regressed != new_regressed:
            changed_regression_status_count += 1

        bucket = str(row.get("passrate"))
        info = bucket_summary.setdefault(
            bucket,
            {
                "count": 0,
                "old_mean_sum": 0.0,
                "new_mean_sum": 0.0,
                "rows_changed": 0,
                "old_regressed": 0,
                "new_regressed": 0,
            },
        )
        info["count"] += 1
        info["old_mean_sum"] += old_mean
        info["new_mean_sum"] += new_mean
        info["rows_changed"] += 1 if row_changed else 0
        info["old_regressed"] += 1 if old_regressed else 0
        info["new_regressed"] += 1 if new_regressed else 0

    for info in bucket_summary.values():
        count = info["count"]
        info["old_mean_solver_accuracy"] = info.pop("old_mean_sum") / count if count else 0.0
        info["new_mean_solver_accuracy"] = info.pop("new_mean_sum") / count if count else 0.0

    summary = {
        "rows_path": args.rows_path,
        "num_rows": len(rows),
        "num_samples": sum(len(row.get("solver_samples") or []) for row in rows),
        "percent_tol": args.percent_tol,
        "changed_sample_count": changed_sample_count,
        "changed_row_count": changed_row_count,
        "changed_regression_status_count": changed_regression_status_count,
        "percent_like_sample_count": percent_like_sample_count,
        "percent_like_changed_sample_count": percent_like_changed_sample_count,
        "bucket_summary": bucket_summary,
    }

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "flipped_examples.json").write_text(json.dumps(flipped_examples, indent=2), encoding="utf-8")

    lines = [
        "# Percent-Relaxed Rescore Check",
        "",
        f"- rows path: `{args.rows_path}`",
        f"- percent tolerance: `{args.percent_tol}`",
        f"- changed samples: `{changed_sample_count}`",
        f"- changed rows: `{changed_row_count}`",
        f"- changed regression status rows: `{changed_regression_status_count}`",
        f"- percent-like samples: `{percent_like_sample_count}`",
        f"- percent-like changed samples: `{percent_like_changed_sample_count}`",
        "",
        "## Bucket Summary",
        "",
    ]
    for bucket in sorted(bucket_summary, key=lambda x: float(x)):
        info = bucket_summary[bucket]
        lines.extend(
            [
                f"### Bucket {bucket}",
                f"- rows: `{info['count']}`",
                f"- old mean solver accuracy: `{info['old_mean_solver_accuracy']:.6f}`",
                f"- new mean solver accuracy: `{info['new_mean_solver_accuracy']:.6f}`",
                f"- rows with any sample changed: `{info['rows_changed']}`",
                f"- old regressed rows: `{info['old_regressed']}`",
                f"- new regressed rows: `{info['new_regressed']}`",
                "",
            ]
        )

    lines.extend(["## Example Flips", ""])
    for ex in flipped_examples[:20]:
        lines.extend(
            [
                f"### Row {ex['row_id']} seed {ex['seed']}",
                f"- ground truth: `{ex['ground_truth']}`",
                f"- predicted: `{ex['predicted_answer']}`",
                f"- old -> new: `{ex['old_score']} -> {ex['new_score']}`",
                "",
            ]
        )
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
