from __future__ import annotations

import argparse
import gc
import json
from datetime import date
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, DatasetDict, load_from_disk
from math_verify import parse, verify
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from process_deepscaler_dataset import extract_answer_from_solution


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = "/workspace/aime2025_amc2023_eval_set"
DEFAULT_ABSTRACTION_MODEL = (
    REPO_ROOT / "merged_models" / "qwen3_1_7b_principle_generator_ckpt1736"
)
DEFAULT_SOLVER_MODEL = "Qwen/Qwen3-1.7B"
DEFAULT_ABSTRACTION_TEMPLATE = (
    REPO_ROOT / "prompt_templates" / "sft_principle_generation.txt"
)
DEFAULT_SOLVER_TEMPLATE = (
    REPO_ROOT / "prompt_templates" / "hint_conditioned_problem_solving_rich_v1.txt"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "outputs"
    / date.today().isoformat()
    / "contrastive_abstraction_prompting"
    / "hierarchical_principle_baseline_qwen3_1_7b_4x4_rich_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET)
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
    )
    parser.add_argument(
        "--abstraction_model",
        type=str,
        default=str(DEFAULT_ABSTRACTION_MODEL),
    )
    parser.add_argument("--solver_model", type=str, default=DEFAULT_SOLVER_MODEL)
    parser.add_argument(
        "--abstraction_prompt_template_path",
        type=str,
        default=str(DEFAULT_ABSTRACTION_TEMPLATE),
    )
    parser.add_argument(
        "--solver_prompt_template_path",
        type=str,
        default=str(DEFAULT_SOLVER_TEMPLATE),
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--num_abstractions", type=int, default=4)
    parser.add_argument("--num_solver_samples", type=int, default=4)
    parser.add_argument("--abstraction_max_tokens", type=int, default=2048)
    parser.add_argument("--solver_max_tokens", type=int, default=32768)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument(
        "--abstraction_max_num_batched_tokens",
        type=int,
        default=8192,
    )
    parser.add_argument(
        "--solver_max_num_batched_tokens",
        type=int,
        default=4096,
    )
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--disable_solver_thinking", action="store_true")
    parser.add_argument("--print_sample_prompts", action="store_true")
    return parser.parse_args()


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def load_prompt_template(path_str: str) -> str:
    return resolve_path(path_str).read_text(encoding="utf-8")


def load_dataset_rows(path_str: str) -> Dataset:
    dataset = load_from_disk(path_str)
    if isinstance(dataset, DatasetDict):
        if "train" in dataset:
            dataset = dataset["train"]
        else:
            dataset = next(iter(dataset.values()))
    assert isinstance(dataset, Dataset)
    return dataset


def select_shard(
    dataset: Dataset,
    shard: int,
    num_shards: int,
    max_rows: int | None,
) -> Dataset:
    if num_shards < 1:
        raise ValueError("num_shards must be >= 1")
    if shard < 0 or shard >= num_shards:
        raise ValueError("shard must satisfy 0 <= shard < num_shards")

    shard_size = dataset.num_rows // num_shards
    start = shard * shard_size
    end = dataset.num_rows if shard == num_shards - 1 else (shard + 1) * shard_size
    selected = dataset.select(range(start, end))
    if max_rows is not None:
        selected = selected.select(range(min(max_rows, selected.num_rows)))
    return selected


def render_template(template: str, problem: str, abstraction: str = "") -> str:
    rendered = template
    replacements = {
        "{{PROBLEM}}": problem,
        "{problem_description}": problem,
        "{{ABSTRACTION}}": abstraction,
        "{cheatsheet}": abstraction,
    }
    for old, new in replacements.items():
        rendered = rendered.replace(old, new)
    return rendered


def apply_chat_template(
    tokenizer: AutoTokenizer,
    prompt: str,
    enable_thinking: bool,
) -> str:
    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def cleanup_model(llm: LLM | None) -> None:
    if llm is not None:
        del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def score_answer(expected_answer: str, predicted_boxed_answer: str | None) -> bool:
    if predicted_boxed_answer is None:
        return False
    try:
        target = parse(f"\\boxed{{{expected_answer}}}")
        predicted = parse(predicted_boxed_answer)
    except Exception:
        return False
    try:
        return bool(verify(target, predicted))
    except Exception:
        return False


def generate_abstractions(
    examples: list[dict[str, Any]],
    prompt_template: str,
    args: argparse.Namespace,
) -> list[list[str]]:
    tokenizer = AutoTokenizer.from_pretrained(args.abstraction_model)
    prompts: list[str] = []
    raw_prompts: list[str] = []
    for example in tqdm(examples, desc="creating abstraction prompts"):
        prompt = render_template(prompt_template, problem=example["problem"])
        raw_prompts.append(prompt)
        prompts.append(apply_chat_template(tokenizer, prompt, enable_thinking=False))

    if args.print_sample_prompts and raw_prompts:
        print("\n[SAMPLE ABSTRACTION PROMPT]\n")
        print(raw_prompts[0])

    llm = LLM(
        model=args.abstraction_model,
        dtype=args.dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_batched_tokens=args.abstraction_max_num_batched_tokens,
    )
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.abstraction_max_tokens,
        n=args.num_abstractions,
        seed=args.seed,
    )
    outputs = llm.generate(prompts, sampling_params)
    abstractions = [[candidate.text for candidate in output.outputs] for output in outputs]
    cleanup_model(llm)
    return abstractions


def generate_solver_rollouts(
    examples: list[dict[str, Any]],
    abstractions: list[list[str]],
    prompt_template: str,
    args: argparse.Namespace,
) -> list[list[list[str]]]:
    tokenizer = AutoTokenizer.from_pretrained(args.solver_model)
    prompts: list[str] = []
    raw_prompts: list[str] = []
    prompt_index: list[tuple[int, int]] = []

    for row_idx, example in enumerate(tqdm(examples, desc="creating solver prompts")):
        for abstraction_idx, abstraction in enumerate(abstractions[row_idx]):
            prompt = render_template(
                prompt_template,
                problem=example["problem"],
                abstraction=abstraction,
            )
            raw_prompts.append(prompt)
            prompts.append(
                apply_chat_template(
                    tokenizer,
                    prompt,
                    enable_thinking=not args.disable_solver_thinking,
                )
            )
            prompt_index.append((row_idx, abstraction_idx))

    if args.print_sample_prompts and raw_prompts:
        print("\n[SAMPLE SOLVER PROMPT]\n")
        print(raw_prompts[0])

    llm = LLM(
        model=args.solver_model,
        dtype=args.dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_batched_tokens=args.solver_max_num_batched_tokens,
    )
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.solver_max_tokens,
        n=args.num_solver_samples,
        seed=args.seed,
    )
    outputs = llm.generate(prompts, sampling_params)

    rollouts: list[list[list[str]]] = [
        [[] for _ in range(args.num_abstractions)] for _ in range(len(examples))
    ]
    for (row_idx, abstraction_idx), output in zip(prompt_index, outputs):
        rollouts[row_idx][abstraction_idx] = [candidate.text for candidate in output.outputs]

    cleanup_model(llm)
    return rollouts


def build_result_rows(
    examples: list[dict[str, Any]],
    abstractions: list[list[str]],
    rollouts: list[list[list[str]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row_idx, example in enumerate(examples):
        answers_by_abstraction: list[list[str | None]] = []
        num_correct_by_abstraction: list[int] = []
        passrate_by_abstraction: list[float] = []
        solved_by_abstraction: list[bool] = []
        flattened_solutions: list[str] = []
        flattened_answers: list[str | None] = []

        for solver_outputs in rollouts[row_idx]:
            extracted_answers = [
                extract_answer_from_solution(solution_text)
                for solution_text in solver_outputs
            ]
            num_correct = sum(
                score_answer(example["answer"], answer)
                for answer in extracted_answers
            )
            answers_by_abstraction.append(extracted_answers)
            num_correct_by_abstraction.append(num_correct)
            passrate_by_abstraction.append(
                num_correct / len(extracted_answers) if extracted_answers else 0.0
            )
            solved_by_abstraction.append(num_correct > 0)
            flattened_solutions.extend(solver_outputs)
            flattened_answers.extend(extracted_answers)

        total_correct = sum(num_correct_by_abstraction)
        total_samples = len(flattened_solutions)
        num_invalid = sum(answer is None for answer in flattened_answers)

        row = dict(example)
        row["row_idx"] = row_idx
        row["generated_abstractions"] = abstractions[row_idx]
        row["abstraction_conditioned_problem_solutions"] = rollouts[row_idx]
        row["generated_answer_by_abstraction"] = answers_by_abstraction
        row["generated_solution"] = flattened_solutions
        row["generated_answer"] = flattened_answers
        row["num_correct_by_abstraction"] = num_correct_by_abstraction
        row["passrate_by_abstraction"] = passrate_by_abstraction
        row["solved_by_abstraction"] = solved_by_abstraction
        row["num_correct"] = total_correct
        row["passrate"] = total_correct / total_samples if total_samples else 0.0
        row["solved"] = total_correct > 0
        row["num_invalid"] = num_invalid
        rows.append(row)
    return rows


def metrics_from_rows(
    rows: list[dict[str, Any]],
    num_abstractions: int,
    num_solver_samples: int,
) -> dict[str, Any]:
    total_rows = len(rows)
    total_samples = sum(len(row["generated_solution"]) for row in rows)
    total_correct_samples = sum(int(row["num_correct"]) for row in rows)
    total_invalid_samples = sum(int(row["num_invalid"]) for row in rows)
    total_abstraction_rollouts = sum(
        len(row["solved_by_abstraction"]) for row in rows
    )
    total_solved_abstractions = sum(
        sum(1 for solved in row["solved_by_abstraction"] if solved)
        for row in rows
    )
    total_solved_problems = sum(1 for row in rows if row["solved"])

    problem_pass_rate = (
        total_solved_problems / total_rows if total_rows else 0.0
    )
    abstraction_pass_rate = (
        total_solved_abstractions / total_abstraction_rollouts
        if total_abstraction_rollouts
        else 0.0
    )
    sample_accuracy = (
        total_correct_samples / total_samples if total_samples else 0.0
    )

    metrics = {
        "num_rows": total_rows,
        "num_abstractions_per_problem": num_abstractions,
        "num_solver_samples_per_abstraction": num_solver_samples,
        "num_solver_samples_per_problem": num_abstractions * num_solver_samples,
        "total_solver_samples": total_samples,
        "total_correct_samples": total_correct_samples,
        "total_invalid_samples": total_invalid_samples,
        "sample_accuracy": sample_accuracy,
        "mean_num_correct_per_problem": (
            total_correct_samples / total_rows if total_rows else 0.0
        ),
        "mean_num_correct_per_abstraction": (
            total_correct_samples / total_abstraction_rollouts
            if total_abstraction_rollouts
            else 0.0
        ),
        "problem_solved_rate": problem_pass_rate,
        "mean_abstraction_solved_rate": abstraction_pass_rate,
    }
    metrics[f"problem_pass_at_{num_abstractions * num_solver_samples}"] = (
        problem_pass_rate
    )
    metrics[f"abstraction_pass_at_{num_solver_samples}"] = abstraction_pass_rate
    return metrics


def per_source_report(
    rows: list[dict[str, Any]],
    num_abstractions: int,
    num_solver_samples: int,
) -> dict[str, dict[str, Any]]:
    if not rows or "source" not in rows[0]:
        return {}
    reports: dict[str, dict[str, Any]] = {}
    for source in sorted({row["source"] for row in rows}):
        source_rows = [row for row in rows if row["source"] == source]
        reports[source] = metrics_from_rows(
            source_rows,
            num_abstractions=num_abstractions,
            num_solver_samples=num_solver_samples,
        )
    return reports


def build_report(
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
    abstraction_prompt_path: Path,
    solver_prompt_path: Path,
) -> dict[str, Any]:
    return {
        "config": {
            "dataset": args.dataset,
            "output_dir": args.output_dir,
            "abstraction_model": args.abstraction_model,
            "solver_model": args.solver_model,
            "abstraction_prompt_template_path": str(abstraction_prompt_path),
            "solver_prompt_template_path": str(solver_prompt_path),
            "seed": args.seed,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "num_abstractions": args.num_abstractions,
            "num_solver_samples": args.num_solver_samples,
            "abstraction_max_tokens": args.abstraction_max_tokens,
            "solver_max_tokens": args.solver_max_tokens,
            "tensor_parallel_size": args.tensor_parallel_size,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "abstraction_max_num_batched_tokens": args.abstraction_max_num_batched_tokens,
            "solver_max_num_batched_tokens": args.solver_max_num_batched_tokens,
            "dtype": args.dtype,
            "shard": args.shard,
            "num_shards": args.num_shards,
            "max_rows": args.max_rows,
            "solver_thinking_enabled": not args.disable_solver_thinking,
        },
        "overall": metrics_from_rows(
            rows,
            num_abstractions=args.num_abstractions,
            num_solver_samples=args.num_solver_samples,
        ),
        "by_source": per_source_report(
            rows,
            num_abstractions=args.num_abstractions,
            num_solver_samples=args.num_solver_samples,
        ),
    }


def format_metric(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def report_to_markdown(report: dict[str, Any]) -> str:
    return '# Abstraction-Conditioned Baseline Evaluation\n\n' + json.dumps(report, indent=2) + '\n'

def save_outputs(
    rows: list[dict[str, Any]],
    report: dict[str, Any],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = Dataset.from_list(rows)
    dataset.save_to_disk(str(output_dir / "scored_dataset"))
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    (output_dir / "REPORT.md").write_text(
        report_to_markdown(report),
        encoding="utf-8",
    )

    sample_rows = []
    for row in rows[: min(5, len(rows))]:
        sample_rows.append(
            {
                "row_idx": row["row_idx"],
                "source": row.get("source"),
                "answer": row["answer"],
                "num_correct": row["num_correct"],
                "passrate": row["passrate"],
                "solved": row["solved"],
                "generated_abstractions": row["generated_abstractions"],
                "generated_answer_by_abstraction": row["generated_answer_by_abstraction"],
            }
        )
    (output_dir / "sample_rows.json").write_text(
        json.dumps(sample_rows, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    print("Args:")
    print(args)

    abstraction_prompt_path = resolve_path(args.abstraction_prompt_template_path)
    solver_prompt_path = resolve_path(args.solver_prompt_template_path)
    abstraction_prompt_template = load_prompt_template(args.abstraction_prompt_template_path)
    solver_prompt_template = load_prompt_template(args.solver_prompt_template_path)

    dataset = load_dataset_rows(args.dataset)
    dataset = select_shard(
        dataset,
        shard=args.shard,
        num_shards=args.num_shards,
        max_rows=args.max_rows,
    )
    examples = [dict(dataset[idx]) for idx in range(dataset.num_rows)]

    if not examples:
        raise ValueError("No dataset rows selected for evaluation")

    if torch.cuda.is_available():
        print(f"Device name = {torch.cuda.get_device_name(0)}")

    abstractions = generate_abstractions(
        examples=examples,
        prompt_template=abstraction_prompt_template,
        args=args,
    )
    rollouts = generate_solver_rollouts(
        examples=examples,
        abstractions=abstractions,
        prompt_template=solver_prompt_template,
        args=args,
    )
    rows = build_result_rows(
        examples=examples,
        abstractions=abstractions,
        rollouts=rollouts,
    )
    report = build_report(
        rows=rows,
        args=args,
        abstraction_prompt_path=abstraction_prompt_path,
        solver_prompt_path=solver_prompt_path,
    )

    output_dir = Path(args.output_dir)
    save_outputs(rows=rows, report=report, output_dir=output_dir)

    print("\nOVERALL REPORT")
    print(json.dumps(report["overall"], indent=2))
    print("\nBY SOURCE")
    print(json.dumps(report["by_source"], indent=2))
    print(f"\nSaved outputs to {output_dir}")


if __name__ == "__main__":
    main()
