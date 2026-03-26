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

PERCENT_WORD_RE = re.compile(r"\bpercent(?:age)?s?\b", flags=re.IGNORECASE)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    return parser


def gt_is_percent_like(text: str) -> bool:
    if not text:
        return False
    return ("%" in text) or (r"\%" in text) or bool(PERCENT_WORD_RE.search(text))


def strip_percent_markers(text: str) -> str:
    text = text.strip()
    text = text.replace(r"\%", "")
    text = text.replace("%", "")
    text = PERCENT_WORD_RE.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def compute_score_gt_percent_only_exact(solution_str: str, ground_truth: str) -> float:
    base = patched_reward.compute_score(None, solution_str, ground_truth)
    if base == 1.0:
        return 1.0

    if not gt_is_percent_like(ground_truth):
        return 0.0

    boxed_answers = patched_reward.extract_boxed_answers(solution_str)
    if not boxed_answers:
        return 0.0

    pred_raw = patched_reward.unbox(boxed_answers[-1])
    gt_raw = ground_truth.strip()
    pred_norm = strip_percent_markers(pred_raw)
    gt_norm = strip_percent_markers(gt_raw)
    if not pred_norm or not gt_norm:
        return 0.0

    pred_boxed = f"\\boxed{{{pred_norm}}}"
    return patched_reward.score_single_answer(pred_boxed, gt_norm)


def main() -> None:
    args = build_parser().parse_args()
    rows = json.loads(Path(args.rows_path).read_text(encoding="utf-8"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    changed_sample_count = 0
    changed_row_count = 0
    changed_regression_status_count = 0
    flipped_examples: list[dict] = []

    for row in rows:
        row_changed = False
        old_scores = []
        new_scores = []
        for sample in row.get("solver_samples") or []:
            old = 1.0 if sample.get("correct") else 0.0
            new = compute_score_gt_percent_only_exact(sample.get("output_text") or "", row.get("answer", ""))
            old_scores.append(old)
            new_scores.append(new)
            if old != new:
                changed_sample_count += 1
                row_changed = True
                if len(flipped_examples) < 500:
                    flipped_examples.append(
                        {
                            "row_id": row.get("row_id"),
                            "seed": sample.get("seed"),
                            "ground_truth": row.get("answer"),
                            "predicted_answer": sample.get("predicted_answer"),
                            "old_score": old,
                            "new_score": new,
                            "passrate": row.get("passrate"),
                            "old_mean_solver_accuracy": row.get("mean_solver_accuracy"),
                        }
                    )
        old_mean = sum(old_scores) / len(old_scores) if old_scores else 0.0
        new_mean = sum(new_scores) / len(new_scores) if new_scores else 0.0
        if row_changed:
            changed_row_count += 1
        if (old_mean < float(row.get("passrate", 0.0))) != (new_mean < float(row.get("passrate", 0.0))):
            changed_regression_status_count += 1

    summary = {
        "rows_path": args.rows_path,
        "num_rows": len(rows),
        "num_samples": sum(len(row.get("solver_samples") or []) for row in rows),
        "changed_sample_count": changed_sample_count,
        "changed_row_count": changed_row_count,
        "changed_regression_status_count": changed_regression_status_count,
        "examples_99_115": {},
    }

    indexed_rows = {row.get("row_id"): row for row in rows}
    for target in (99, 115):
        row = indexed_rows[target]
        samples = []
        for sample in row.get("solver_samples") or []:
            old = 1.0 if sample.get("correct") else 0.0
            new = compute_score_gt_percent_only_exact(sample.get("output_text") or "", row.get("answer", ""))
            samples.append(
                {
                    "seed": sample.get("seed"),
                    "pred": sample.get("predicted_answer"),
                    "old": old,
                    "new": new,
                }
            )
        summary["examples_99_115"][str(target)] = samples

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "flipped_examples.json").write_text(json.dumps(flipped_examples, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
