from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import signal
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any, Optional

from datasets import Dataset, DatasetDict, load_from_disk
from gepa.optimize_anything import GEPAConfig, EngineConfig, ReflectionConfig, optimize_anything

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.contrastive_abstraction_utils import repo_path, strip_think_blocks


DEFAULT_DATASET = "contrastive_abstraction_datasets/deepscaler_passrate_gt0_3k"
DEFAULT_BASE_MODEL = "Qwen/Qwen3-1.7B"
DEFAULT_PRINCIPLE_PROMPT = "prompt_templates/principle_extraction_template_v5.txt"
DEFAULT_SOLVER_PROMPT = "prompt_templates/hint_conditioned_problem_solving_rich_v1.txt"
DEFAULT_REFLECTION_MODEL = "openai/gpt-5-mini"
PASSRATE_BUCKETS = ("0.25", "0.50", "0.75", "1.00")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed_prompt_path", type=str, default=DEFAULT_PRINCIPLE_PROMPT)
    parser.add_argument("--dataset_path", type=str, default=DEFAULT_DATASET)
    parser.add_argument("--solver_prompt_path", type=str, default=DEFAULT_SOLVER_PROMPT)
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--train_size", type=int, default=128)
    parser.add_argument("--val_size", type=int, default=32)
    parser.add_argument("--train_bucket_counts", type=str, default="")
    parser.add_argument("--val_bucket_counts", type=str, default="")
    parser.add_argument("--split_seed", type=int, default=0)
    parser.add_argument("--reflection_model", type=str, default=DEFAULT_REFLECTION_MODEL)
    parser.add_argument("--max_metric_calls", type=int, default=800)
    parser.add_argument("--base_model", type=str, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--principle_temperature", type=float, default=0.0)
    parser.add_argument("--solver_temperature", type=float, default=0.6)
    parser.add_argument("--solver_seeds", type=str, default="1001,1002,1003,1004")
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--max_solution_chars", type=int, default=6000)
    parser.add_argument("--max_model_len", type=int, default=32768)
    parser.add_argument("--max_num_batched_tokens", type=int, default=4096)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--prompt_batch_size", type=int, default=64)
    return parser.parse_args()


def resolve_repo_file(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return repo_path(path_str)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any] | list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def candidate_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def load_dataset_dict(path: str) -> DatasetDict:
    loaded = load_from_disk(str(resolve_repo_file(path)))
    if isinstance(loaded, DatasetDict):
        return loaded
    return DatasetDict({"train": loaded})


def ensure_trace_pairs(example: dict[str, Any]) -> tuple[list[tuple[int, str]], list[tuple[int, str]], list[bool]]:
    generated_solutions = example.get("generated_solution")
    generated_answers = example.get("generated_answer")
    if generated_solutions is not None and generated_answers is not None:
        solutions = list(generated_solutions)
        if example.get("solution_correctness") is not None:
            correctness = [bool(value) for value in example["solution_correctness"]]
        else:
            raise ValueError("solution_correctness is required when generated_solution/generated_answer are present")
        correct_pairs = [(idx, solutions[idx]) for idx, is_correct in enumerate(correctness) if is_correct]
        incorrect_pairs = [(idx, solutions[idx]) for idx, is_correct in enumerate(correctness) if not is_correct]
        return correct_pairs, incorrect_pairs, correctness

    correct_indices = list(example.get("selected_correct_indices") or range(len(example.get("selected_correct_solutions") or [])))
    incorrect_indices = list(
        example.get("selected_incorrect_indices") or range(len(example.get("selected_incorrect_solutions") or []))
    )
    correct_solutions = list(example.get("selected_correct_solutions") or [])
    incorrect_solutions = list(example.get("selected_incorrect_solutions") or [])
    correct_pairs = list(zip(correct_indices, correct_solutions))
    incorrect_pairs = list(zip(incorrect_indices, incorrect_solutions))
    correctness = [True] * len(correct_pairs) + [False] * len(incorrect_pairs)
    return correct_pairs, incorrect_pairs, correctness


def first_nonempty_trace(pairs: list[tuple[int, str]]) -> tuple[Optional[int], str]:
    for trace_idx, trace_text in pairs:
        cleaned = strip_think_blocks(trace_text)
        if cleaned:
            return trace_idx, cleaned
    return None, ""


def passrate_bucket(passrate: float) -> str:
    if math.isclose(passrate, 1.0):
        return "1.00"
    if math.isclose(passrate, 0.75):
        return "0.75"
    if math.isclose(passrate, 0.5):
        return "0.50"
    return "0.25"


def normalize_bucket_label(raw: str) -> str:
    label = raw.strip()
    mapping = {
        "0.25": "0.25",
        ".25": "0.25",
        "0.50": "0.50",
        "0.5": "0.50",
        ".5": "0.50",
        "0.75": "0.75",
        ".75": "0.75",
        "1": "1.00",
        "1.0": "1.00",
        "1.00": "1.00",
    }
    if label not in mapping:
        raise ValueError(f"unsupported passrate bucket label: {raw}")
    return mapping[label]


def bucket_seed(seed: int, bucket: str, label: str) -> int:
    digest = hashlib.sha256(f"{seed}:{label}:{bucket}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def allocate_counts(bucket_to_rows: dict[str, list[dict[str, Any]]], target_total: int) -> dict[str, int]:
    total_available = sum(len(rows) for rows in bucket_to_rows.values())
    if target_total > total_available:
        raise ValueError(f"requested {target_total} rows but only {total_available} eligible rows are available")

    exact = {bucket: target_total * len(rows) / total_available for bucket, rows in bucket_to_rows.items()}
    counts = {bucket: min(len(bucket_to_rows[bucket]), int(math.floor(value))) for bucket, value in exact.items()}
    remainder = target_total - sum(counts.values())

    order = sorted(
        PASSRATE_BUCKETS,
        key=lambda bucket: (exact[bucket] - counts[bucket], len(bucket_to_rows[bucket]), bucket),
        reverse=True,
    )
    for bucket in order:
        if remainder <= 0:
            break
        if counts[bucket] < len(bucket_to_rows[bucket]):
            counts[bucket] += 1
            remainder -= 1

    if remainder > 0:
        for bucket in PASSRATE_BUCKETS:
            while remainder > 0 and counts[bucket] < len(bucket_to_rows[bucket]):
                counts[bucket] += 1
                remainder -= 1

    if sum(counts.values()) != target_total:
        raise ValueError("failed to allocate the requested number of rows")
    return counts


def parse_bucket_counts(
    raw: str,
    target_total: int,
    available_rows: dict[str, list[dict[str, Any]]],
    label: str,
) -> dict[str, int]:
    counts = {bucket: 0 for bucket in PASSRATE_BUCKETS}
    if not raw.strip():
        return allocate_counts(available_rows, target_total)

    for item in raw.split(","):
        piece = item.strip()
        if not piece:
            continue
        bucket_raw, count_raw = piece.split("=", 1)
        bucket = normalize_bucket_label(bucket_raw)
        counts[bucket] = int(count_raw.strip())

    if sum(counts.values()) != target_total:
        raise ValueError(
            f"{label} bucket counts sum to {sum(counts.values())}, expected {target_total}"
        )

    for bucket in PASSRATE_BUCKETS:
        available = len(available_rows[bucket])
        if counts[bucket] > available:
            raise ValueError(
                f"requested {counts[bucket]} rows for bucket {bucket} in {label}, but only {available} are available"
            )
    return counts


def build_split_manifest(args: argparse.Namespace) -> dict[str, Any]:
    dataset_dict = load_dataset_dict(args.dataset_path)
    bucket_to_rows: dict[str, list[dict[str, Any]]] = {bucket: [] for bucket in PASSRATE_BUCKETS}
    eligible_rows = 0

    for split_name, split_ds in dataset_dict.items():
        assert isinstance(split_ds, Dataset)
        for idx, example in enumerate(split_ds):
            passrate = float(example.get("passrate", 0.0))
            if passrate <= 0.0:
                continue

            correct_pairs, _, _ = ensure_trace_pairs(example)
            first_correct_idx, first_correct_trace = first_nonempty_trace(correct_pairs)
            if not first_correct_trace:
                continue

            row = {
                "row_id": int(example.get("row_id", idx)),
                "problem": example["problem"],
                "answer": example["answer"],
                "passrate": passrate,
                "num_correct": int(example.get("num_correct", len(correct_pairs))),
                "source_split": split_name,
                "first_correct_trace_index": first_correct_idx,
                "first_correct_trace_text": first_correct_trace,
            }
            bucket_to_rows[passrate_bucket(passrate)].append(row)
            eligible_rows += 1

    if eligible_rows < args.train_size + args.val_size:
        raise ValueError(
            f"only {eligible_rows} eligible rows after filtering, need {args.train_size + args.val_size}"
        )

    train_counts = parse_bucket_counts(args.train_bucket_counts, args.train_size, bucket_to_rows, "train")
    remaining_rows = {
        bucket: bucket_to_rows[bucket][train_counts[bucket] :]
        for bucket in PASSRATE_BUCKETS
    }
    val_counts = parse_bucket_counts(args.val_bucket_counts, args.val_size, remaining_rows, "val")

    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []

    for bucket in PASSRATE_BUCKETS:
        rows = sorted(bucket_to_rows[bucket], key=lambda row: row["row_id"])
        rng = random.Random(bucket_seed(args.split_seed, bucket, "sample"))
        rng.shuffle(rows)
        train_rows.extend(rows[: train_counts[bucket]])
        val_rows.extend(rows[train_counts[bucket] : train_counts[bucket] + val_counts[bucket]])

    random.Random(bucket_seed(args.split_seed, "train", "order")).shuffle(train_rows)
    random.Random(bucket_seed(args.split_seed, "val", "order")).shuffle(val_rows)

    bucket_summary = {
        bucket: {
            "eligible": len(bucket_to_rows[bucket]),
            "train": train_counts[bucket],
            "val": val_counts[bucket],
        }
        for bucket in PASSRATE_BUCKETS
    }

    return {
        "source_dataset_path": args.dataset_path,
        "split_seed": args.split_seed,
        "train_size": len(train_rows),
        "val_size": len(val_rows),
        "train_bucket_counts": train_counts,
        "val_bucket_counts": val_counts,
        "bucket_summary": bucket_summary,
        "train_rows": train_rows,
        "val_rows": val_rows,
    }


def has_prompt_echo_instructions(candidate: str) -> str | None:
    lowered = candidate.lower()
    forbidden_phrases = [
        "followed by the problem and solution traces blocks",
        "include the original problem and the selected trace block",
        "include the original problem and the selected trace",
        "include the problem and solution traces blocks",
        "then include the original problem",
        "after the list, include the original problem",
        "after the list, include",
        "problem / solution traces blocks",
        "required problem / solution traces blocks",
        "literal blocks (do not modify)",
    ]
    for phrase in forbidden_phrases:
        if phrase in lowered:
            return phrase
    return None


def run_dir_has_output_leakage(run_dir: Path) -> bool:
    rows_path = run_dir / "rows.json"
    if not rows_path.exists():
        return False
    rows = read_json(rows_path)
    for row in rows:
        for key in ("generated_principles_text", "principles"):
            text = row.get(key, "")
            if isinstance(text, str) and (
                "PROBLEM:" in text or "SOLUTION TRACES:" in text
            ):
                return True
    return False


def validate_candidate(candidate: str) -> tuple[bool, str]:
    required_placeholders = ["{{PROBLEM}}", "{{CORRECT_SOLUTIONS_BLOCK}}"]
    missing = [placeholder for placeholder in required_placeholders if placeholder not in candidate]
    if missing:
        return False, f"missing placeholders: {missing}"
    if not candidate.strip():
        return False, "candidate prompt is empty"
    echo_phrase = has_prompt_echo_instructions(candidate)
    if echo_phrase is not None:
        return False, f"candidate instructs the extractor to echo problem/trace content: {echo_phrase}"
    return True, ""


def cleanup_orphan_vllm_engine_cores() -> list[int]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    killed: list[int] = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",", 1)]
        if len(parts) != 2:
            continue
        pid_str, process_name = parts
        if process_name != "VLLM::EngineCore":
            continue
        if not pid_str.isdigit():
            continue
        pid = int(pid_str)
        try:
            os.kill(pid, signal.SIGTERM)
            killed.append(pid)
        except ProcessLookupError:
            continue
    return killed


def maybe_run_candidate_eval(candidate: str) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    args = run_eval.args
    output_root = resolve_repo_file(args.output_root)
    prompts_dir = output_root / "candidate_prompts"
    evals_dir = output_root / "candidate_evals"
    prompts_dir.mkdir(parents=True, exist_ok=True)
    evals_dir.mkdir(parents=True, exist_ok=True)

    h = candidate_hash(candidate)
    prompt_path = prompts_dir / f"{h}.txt"
    run_dir = evals_dir / h
    scores_path = run_dir / "scores_by_row.json"
    report_path = run_dir / "report.json"

    if not scores_path.exists() or not report_path.exists():
        prompt_path.write_text(candidate, encoding="utf-8")
        eval_cmd = [
            "/workspace/miniconda3/bin/conda",
            "run",
            "--no-capture-output",
            "-n",
            "abstraction",
            "python",
            "scripts/eval_principle_extraction_prompt.py",
            "--rows_manifest_path",
            str(resolve_repo_file(args.output_root) / "split_manifest.json"),
            "--candidate_prompt_path",
            str(prompt_path),
            "--output_dir",
            str(run_dir),
            "--base_model",
            args.base_model,
            "--solver_prompt_path",
            args.solver_prompt_path,
            "--principle_temperature",
            str(args.principle_temperature),
            "--solver_temperature",
            str(args.solver_temperature),
            "--solver_seeds",
            args.solver_seeds,
            "--max_tokens",
            str(args.max_tokens),
            "--max_solution_chars",
            str(args.max_solution_chars),
            "--max_model_len",
            str(args.max_model_len),
            "--max_num_batched_tokens",
            str(args.max_num_batched_tokens),
            "--gpu_memory_utilization",
            str(args.gpu_memory_utilization),
            "--prompt_batch_size",
            str(args.prompt_batch_size),
        ]
        last_error: subprocess.CalledProcessError | None = None
        for attempt in range(2):
            try:
                subprocess.run(eval_cmd, cwd=REPO_ROOT, check=True)
                last_error = None
                break
            except subprocess.CalledProcessError as exc:
                last_error = exc
                if attempt == 0:
                    cleanup_orphan_vllm_engine_cores()
                    continue
                raise
        if last_error is not None:
            raise last_error

    report = read_json(report_path)
    scores = read_json(scores_path)
    return run_dir, report, scores


def run_eval(candidate: str, example: Any, opt_state: Any = None) -> tuple[float, dict[str, Any]]:
    del opt_state
    valid, reason = validate_candidate(candidate)
    if not valid:
        side_info = {
            "scores": {"row_mean_accuracy": 0.0},
            "Summary": f"invalid candidate: {reason}",
            "InvalidCandidate": True,
        }
        return 0.0, side_info

    if example is None:
        raise ValueError("example must be provided for row-aware GEPA optimization")

    run_dir, report, scores = maybe_run_candidate_eval(candidate)
    if run_dir_has_output_leakage(run_dir):
        side_info = {
            "scores": {"row_mean_accuracy": 0.0},
            "Summary": "invalid candidate: extractor output leaked problem/trace blocks into solver input",
            "InvalidCandidate": True,
            "RunDir": str(run_dir),
        }
        return 0.0, side_info

    row_entry = scores[str(example["row_id"])]
    score = float(row_entry["score"])
    side_info = {
        "scores": {
            "row_mean_accuracy": score,
            "train_accuracy": report["train_accuracy"],
            "val_accuracy": report["val_accuracy"],
        },
        "Summary": (
            f"row {example['row_id']} ({row_entry['split']}) mean solve={score:.2f}; "
            f"train={report['train_accuracy']:.4f}; val={report['val_accuracy']:.4f}"
        ),
        "Passrate": example["passrate"],
        "ProblemExcerpt": example["problem"][:240],
        "PrinciplesExcerpt": row_entry.get("principles_excerpt"),
        "PredictedAnswers": row_entry.get("predicted_answers"),
        "RunDir": str(run_dir),
    }
    return score, side_info


def main() -> None:
    args = parse_args()
    run_eval.args = args

    output_root = resolve_repo_file(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY must be set for GEPA reflection.")

    manifest = build_split_manifest(args)
    write_json(output_root / "split_manifest.json", manifest)
    write_json(
        output_root / "split_summary.json",
        {
            "source_dataset_path": manifest["source_dataset_path"],
            "split_seed": manifest["split_seed"],
            "train_size": manifest["train_size"],
            "val_size": manifest["val_size"],
            "train_bucket_counts": manifest["train_bucket_counts"],
            "val_bucket_counts": manifest["val_bucket_counts"],
            "bucket_summary": manifest["bucket_summary"],
        },
    )

    seed_prompt = resolve_repo_file(args.seed_prompt_path).read_text(encoding="utf-8")
    objective = (
        "Maximize downstream solve accuracy on held-out validation rows by improving the principle-extraction prompt. "
        "The extracted principles are passed directly to the solver as the abstraction. "
        "Each row is scored by mean correctness over 4 fixed solver samples."
    )
    background = textwrap.dedent(
        f"""\
        Domain: principle extraction from math solution traces.
        Data: {manifest['train_size']} train rows and {manifest['val_size']} validation rows drawn from {args.dataset_path}.
        Train bucket counts: {manifest['train_bucket_counts']}.
        Validation bucket counts: {manifest['val_bucket_counts']}.
        Trace policy: use only the first non-empty correct trace after stripping <think> blocks.
        Generation policy: principle extraction uses temperature={args.principle_temperature} and only the prompt placeholders {{{{PROBLEM}}}} and {{{{CORRECT_SOLUTIONS_BLOCK}}}}.
        Solver policy: use fixed seeds {args.solver_seeds} at temperature={args.solver_temperature} with the fixed solver prompt {args.solver_prompt_path}; score each row by the mean correctness across those seeds.
        Known failure modes to avoid:
        - broad subject-area labels instead of the decisive reduction
        - long background/theory sections that bury the actionable lever
        - problem-specific clutter or copied calculations
        - unsupported generalizations that are not justified by the trace
        - prompt rewrites that drop the required placeholders
        The prompt should still ask for a small set of reusable principles that are faithful to the trace and useful as a solver hint.
        """
    ).strip()

    config = GEPAConfig(
        engine=EngineConfig(
            run_dir=str(output_root / "gepa_run"),
            max_metric_calls=args.max_metric_calls,
            display_progress_bar=False,
            parallel=False,
            max_workers=1,
            cache_evaluation=True,
            cache_evaluation_storage="disk",
            best_example_evals_k=5,
        ),
        reflection=ReflectionConfig(
            reflection_lm=args.reflection_model,
            skip_perfect_score=False,
        ),
        refiner=None,
        merge=None,
    )

    result = optimize_anything(
        seed_candidate=seed_prompt,
        evaluator=run_eval,
        dataset=manifest["train_rows"],
        valset=manifest["val_rows"],
        objective=objective,
        background=background,
        config=config,
    )

    best_idx = result.best_idx
    summary = {
        "best_idx": best_idx,
        "best_score": result.val_aggregate_scores[best_idx],
        "num_candidates": result.num_candidates,
        "total_metric_calls": result.total_metric_calls,
        "run_dir": result.run_dir,
        "best_candidate": result.best_candidate,
        "all_scores": result.val_aggregate_scores,
        "reflection_model": args.reflection_model,
        "seed_prompt_path": args.seed_prompt_path,
        "dataset_path": args.dataset_path,
        "train_size": manifest["train_size"],
        "val_size": manifest["val_size"],
        "train_bucket_counts": manifest["train_bucket_counts"],
        "val_bucket_counts": manifest["val_bucket_counts"],
    }
    write_json(output_root / "gepa_summary.json", summary)
    (output_root / "gepa_best_prompt.txt").write_text(result.best_candidate, encoding="utf-8")
    write_json(output_root / "gepa_result.json", result.to_dict())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
