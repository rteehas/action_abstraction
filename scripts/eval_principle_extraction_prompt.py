from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Optional

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from process_deepscaler_dataset import extract_answer_from_solution, parse, verify
try:
    from contrastive_abstraction_utils import format_solution_block, read_text, repo_path
except ModuleNotFoundError:
    from scripts.contrastive_abstraction_utils import format_solution_block, read_text, repo_path


DEFAULT_BASE_MODEL = "Qwen/Qwen3-1.7B"
DEFAULT_SOLVER_PROMPT = "prompt_templates/hint_conditioned_problem_solving_rich_v1.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows_manifest_path", type=str, required=True)
    parser.add_argument("--candidate_prompt_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--base_model", type=str, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--solver_prompt_path", type=str, default=DEFAULT_SOLVER_PROMPT)
    parser.add_argument("--principle_temperature", type=float, default=0.0)
    parser.add_argument("--solver_temperature", type=float, default=0.6)
    parser.add_argument("--solver_seeds", type=str, default="1001,1002,1003,1004")
    parser.add_argument("--seed", type=int, default=0)
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


def parse_solver_seeds(raw: str) -> list[int]:
    seeds = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not seeds:
        raise ValueError("solver_seeds must contain at least one integer seed")
    return seeds


def render_principle_prompt(problem: str, first_correct_trace: str, template: str, max_solution_chars: int) -> str:
    correct_block = format_solution_block("Correct Trace", [first_correct_trace], max_chars=max_solution_chars)
    return template.replace("{{PROBLEM}}", problem).replace("{{CORRECT_SOLUTIONS_BLOCK}}", correct_block)


def apply_chat_template(tokenizer: AutoTokenizer, prompts: list[str], enable_thinking: bool) -> list[str]:
    return [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        for prompt in prompts
    ]


def generate_texts(
    llm: LLM,
    tokenizer: AutoTokenizer,
    prompts: list[str],
    sampling: SamplingParams,
    batch_size: int,
    enable_thinking: bool,
) -> list[str]:
    outputs: list[str] = []
    total = len(prompts)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        message_batch = apply_chat_template(tokenizer, prompts[start:end], enable_thinking=enable_thinking)
        batch_outputs = llm.generate(message_batch, sampling)
        outputs.extend(output.outputs[0].text for output in batch_outputs)
        print(f"Generated {end}/{total} prompts")
    return outputs


def score_solver_output(expected_answer: str, solver_text: Optional[str]) -> tuple[Optional[str], bool]:
    boxed = extract_answer_from_solution(solver_text) if solver_text else None
    target = f"\\boxed{{{expected_answer}}}"
    if boxed is None:
        return None, False
    try:
        return boxed, bool(verify(parse(target), parse(boxed)))
    except Exception:
        return boxed, False


def summarize_report(rows: list[dict[str, Any]], solver_seeds: list[int], args: argparse.Namespace) -> dict[str, Any]:
    train_scores = [row["mean_solver_accuracy"] for row in rows if row["split"] == "train"]
    val_scores = [row["mean_solver_accuracy"] for row in rows if row["split"] == "val"]
    principle_lengths = [len((row.get("principles") or "").split()) for row in rows]
    train_baseline = [float(row["passrate"]) for row in rows if row["split"] == "train"]
    val_baseline = [float(row["passrate"]) for row in rows if row["split"] == "val"]

    return {
        "title": "Principle extraction prompt evaluation",
        "rows_manifest_path": args.rows_manifest_path,
        "candidate_prompt_path": args.candidate_prompt_path,
        "solver_prompt_path": args.solver_prompt_path,
        "base_model": args.base_model,
        "principle_temperature": args.principle_temperature,
        "solver_temperature": args.solver_temperature,
        "solver_seeds": solver_seeds,
        "num_rows": len(rows),
        "num_train_rows": len(train_scores),
        "num_val_rows": len(val_scores),
        "train_accuracy": mean(train_scores) if train_scores else 0.0,
        "val_accuracy": mean(val_scores) if val_scores else 0.0,
        "overall_accuracy": mean([row["mean_solver_accuracy"] for row in rows]) if rows else 0.0,
        "train_baseline_passrate_mean": mean(train_baseline) if train_baseline else 0.0,
        "val_baseline_passrate_mean": mean(val_baseline) if val_baseline else 0.0,
        "mean_principle_words": mean(principle_lengths) if principle_lengths else 0.0,
    }


def write_summary_md(path: Path, report: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    lines = [
        f"# {report['title']}",
        "",
        f"- rows manifest: `{report['rows_manifest_path']}`",
        f"- candidate prompt: `{report['candidate_prompt_path']}`",
        f"- base model: `{report['base_model']}`",
        f"- train accuracy: {report['train_accuracy']:.4f}",
        f"- val accuracy: {report['val_accuracy']:.4f}",
        f"- overall accuracy: {report['overall_accuracy']:.4f}",
        f"- train baseline passrate mean: {report['train_baseline_passrate_mean']:.4f}",
        f"- val baseline passrate mean: {report['val_baseline_passrate_mean']:.4f}",
        f"- mean principle words: {report['mean_principle_words']:.2f}",
        f"- solver seeds: {', '.join(str(seed) for seed in report['solver_seeds'])}",
        "",
        "## Sample Rows",
        "",
    ]

    for row in rows[: min(5, len(rows))]:
        lines.extend(
            [
                f"### Row {row['row_id']} ({row['split']})",
                row["problem"],
                "",
                "#### Principles",
                row.get("principles") or "(empty)",
                "",
                "#### Mean Solver Accuracy",
                f"{row['mean_solver_accuracy']:.2f}",
                "",
            ]
        )

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = resolve_repo_file(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = read_json(resolve_repo_file(args.rows_manifest_path))
    candidate_template = read_text(resolve_repo_file(args.candidate_prompt_path))
    solver_template = read_text(resolve_repo_file(args.solver_prompt_path))
    solver_seeds = parse_solver_seeds(args.solver_seeds)

    rows: list[dict[str, Any]] = []
    for split_name in ("train_rows", "val_rows"):
        split = "train" if split_name == "train_rows" else "val"
        for row in manifest[split_name]:
            copied = dict(row)
            copied["split"] = split
            rows.append(copied)

    principle_prompts = [
        render_principle_prompt(
            row["problem"],
            row["first_correct_trace_text"],
            candidate_template,
            max_solution_chars=args.max_solution_chars,
        )
        for row in rows
    ]

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    llm_kwargs = {
        "model": args.base_model,
        "enable_lora": False,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "gpu_memory_utilization": args.gpu_memory_utilization,
    }
    if args.max_model_len is not None:
        llm_kwargs["max_model_len"] = args.max_model_len

    print(f"Device name = {torch.cuda.get_device_name(0)}")
    llm = LLM(**llm_kwargs)

    principle_sampling = SamplingParams(
        temperature=args.principle_temperature,
        top_p=1.0 if args.principle_temperature == 0.0 else 0.95,
        top_k=-1 if args.principle_temperature == 0.0 else 20,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )
    principle_outputs = generate_texts(
        llm,
        tokenizer,
        principle_prompts,
        principle_sampling,
        batch_size=args.prompt_batch_size,
        enable_thinking=False,
    )

    solver_prompts: list[str] = []
    prompt_row_indices: list[int] = []
    for row_idx, (row, principle_prompt, principle_text) in enumerate(zip(rows, principle_prompts, principle_outputs)):
        row["principle_prompt_text"] = principle_prompt
        row["generated_principles_text"] = principle_text
        row["principles"] = principle_text.strip()
        row["abstraction"] = row["principles"]
        row["solver_samples"] = []
        if not row["abstraction"]:
            row["mean_solver_accuracy"] = 0.0
            continue
        solver_prompts.append(
            solver_template.replace("{{PROBLEM}}", row["problem"]).replace("{{ABSTRACTION}}", row["abstraction"])
        )
        prompt_row_indices.append(row_idx)

    for solver_seed in solver_seeds:
        if not solver_prompts:
            break
        solver_sampling = SamplingParams(
            temperature=args.solver_temperature,
            top_p=0.95,
            top_k=20,
            max_tokens=args.max_tokens,
            seed=solver_seed,
        )
        solver_outputs = generate_texts(
            llm,
            tokenizer,
            solver_prompts,
            solver_sampling,
            batch_size=args.prompt_batch_size,
            enable_thinking=False,
        )
        for row_idx, output_text in zip(prompt_row_indices, solver_outputs):
            predicted_answer, correct = score_solver_output(rows[row_idx]["answer"], output_text)
            rows[row_idx]["solver_samples"].append(
                {
                    "seed": solver_seed,
                    "output_text": output_text,
                    "predicted_answer": predicted_answer,
                    "correct": correct,
                }
            )

    scores_by_row: dict[str, Any] = {}
    for row in rows:
        samples = row.get("solver_samples") or []
        row["mean_solver_accuracy"] = (
            sum(1.0 for sample in samples if sample.get("correct")) / len(samples) if samples else 0.0
        )
        scores_by_row[str(row["row_id"])] = {
            "score": row["mean_solver_accuracy"],
            "split": row["split"],
            "passrate": row["passrate"],
            "correct_count": sum(1 for sample in samples if sample.get("correct")),
            "num_samples": len(samples),
            "predicted_answers": [sample.get("predicted_answer") for sample in samples],
            "principles_excerpt": (row.get("principles") or "")[:280],
        }

    report = summarize_report(rows, solver_seeds, args)
    write_json(output_dir / "report.json", report)
    write_json(output_dir / "rows.json", rows)
    write_json(output_dir / "scores_by_row.json", scores_by_row)
    write_summary_md(output_dir / "summary.md", report, rows)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
