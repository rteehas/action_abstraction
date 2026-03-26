from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Optional

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from datasets import Dataset, DatasetDict, load_from_disk
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from contrastive_abstraction_utils import (
    extract_abstraction,
    format_solution_block,
    read_text,
    repo_path,
)
from process_deepscaler_dataset import extract_answer_from_solution, parse, verify


DEFAULT_DATASET = "contrastive_abstraction_datasets/deepscaler_mixed_correct_incorrect"
DEFAULT_LABEL_PROMPT = "prompt_templates/contrastive_abstraction_labeling_prompt.txt"
DEFAULT_SOLVER_PROMPT = "prompt_templates/hint_conditioned_problem_solving.txt"
DEFAULT_FEW_SHOTS = "prompt_templates/contrastive_abstraction_labeling_few_shots.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default=DEFAULT_DATASET)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--max_rows", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--abstraction_lora_path", type=str, default="")
    parser.add_argument("--solver_lora_path", type=str, default="")
    parser.add_argument("--judge_lora_path", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="outputs/2026-03-17/contrastive_abstraction_eval")
    parser.add_argument("--report_title", type=str, default="Contrastive Abstraction Evaluation")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max_tokens", type=int, default=32768)
    parser.add_argument("--max_solution_chars", type=int, default=6000)
    parser.add_argument("--max_model_len", type=int, default=None)
    parser.add_argument("--max_num_batched_tokens", type=int, default=4096)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--label_mode", choices=["contrastive", "problem_only"], default="contrastive")
    parser.add_argument("--abstraction_prompt_path", type=str, default=DEFAULT_LABEL_PROMPT)
    parser.add_argument("--few_shots_path", type=str, default=DEFAULT_FEW_SHOTS)
    parser.add_argument("--solver_prompt_path", type=str, default=DEFAULT_SOLVER_PROMPT)
    parser.add_argument("--label_enable_thinking", action="store_true")
    parser.add_argument("--skip_solver", action="store_true")
    parser.add_argument("--compute_adherence", action="store_true")
    return parser.parse_args()


def resolve_repo_file(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return repo_path(path_str)


def load_split(path: str, split_name: str, max_rows: Optional[int]) -> Dataset:
    dataset = load_from_disk(path)
    if isinstance(dataset, DatasetDict):
        ds = dataset[split_name]
    else:
        ds = dataset
    if max_rows is not None:
        max_rows = min(max_rows, ds.num_rows)
        ds = ds.select(range(max_rows))
    return ds


def generate_with_optional_lora(llm: LLM, prompts: list[str], sampling: SamplingParams, lora_path: str, request_name: str) -> list[str]:
    if lora_path:
        request = LoRARequest(request_name, 1, lora_path)
        outputs = llm.generate(prompts, sampling, lora_request=request)
    else:
        outputs = llm.generate(prompts, sampling)
    return [output.outputs[0].text for output in outputs]


def render_label_prompt(example: dict, template: str, few_shots: str, label_mode: str, max_solution_chars: int) -> str:
    prompt = template.replace("{{FEW_SHOT_EXAMPLES}}", few_shots)
    prompt = prompt.replace("{{PROBLEM}}", example["problem"])
    if label_mode == "contrastive":
        correct_block = format_solution_block(
            "Correct Trace",
            example["selected_correct_solutions"],
            max_chars=max_solution_chars,
        )
        incorrect_block = format_solution_block(
            "Incorrect Trace",
            example["selected_incorrect_solutions"],
            max_chars=max_solution_chars,
        )
        prompt = prompt.replace("{{CORRECT_SOLUTIONS_BLOCK}}", correct_block)
        prompt = prompt.replace("{{INCORRECT_SOLUTIONS_BLOCK}}", incorrect_block)
    else:
        prompt = prompt.replace("{{CORRECT_SOLUTIONS_BLOCK}}", "")
        prompt = prompt.replace("{{INCORRECT_SOLUTIONS_BLOCK}}", "")
    return prompt


def make_solver_prompt(problem: str, abstraction: str, template: str) -> str:
    return template.replace("{{PROBLEM}}", problem).replace("{{ABSTRACTION}}", abstraction)


def make_judge_prompt(problem: str, abstraction: str, solution_text: str) -> str:
    template = read_text(resolve_repo_file("prompt_templates/abstraction_binary_judge_template_updated.txt"))
    return template.replace("{{PROBLEM}}", problem).replace("{{ABSTRACTION}}", abstraction).replace("{{SOLUTION}}", solution_text)


def parse_judge_output(text: str) -> Optional[float]:
    raw = re.search(r"<judgement>([\s\S]*?)</judgement>", text)
    if raw is None:
        return None
    match = raw.group(1).strip()
    if match not in {"0", "1"}:
        return None
    return float(match)


def compute_accuracy(rows: list[dict]) -> dict:
    total = len(rows)
    correct = sum(1 for row in rows if row.get("solver_correct"))
    valid = sum(1 for row in rows if row.get("solver_predicted_answer") is not None)
    adherence_values = [row["adherence"] for row in rows if row.get("adherence") is not None]
    mean_baseline = (sum(row["baseline_passrate"] for row in rows) / total) if total else 0.0
    conditioned = (correct / total) if total else 0.0
    return {
        "num_rows": total,
        "num_correct": correct,
        "accuracy": conditioned,
        "num_valid_answers": valid,
        "baseline_passrate_mean": mean_baseline,
        "baseline_num_correct_mean": (sum(row["baseline_num_correct"] for row in rows) / total) if total else 0.0,
        "absolute_accuracy_lift": conditioned - mean_baseline,
        "relative_accuracy_lift": ((conditioned - mean_baseline) / mean_baseline) if mean_baseline else None,
        "adherence_mean": (sum(adherence_values) / len(adherence_values)) if adherence_values else None,
        "solve_and_adhere_rate": (
            sum(1 for row in rows if row.get("solver_correct") and row.get("adherence") == 1.0) / total
        )
        if adherence_values
        else None,
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_split(args.dataset_path, args.split, args.max_rows)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    label_template = read_text(resolve_repo_file(args.abstraction_prompt_path))
    few_shots = read_text(resolve_repo_file(args.few_shots_path)) if args.few_shots_path else ""
    solver_template = read_text(resolve_repo_file(args.solver_prompt_path))

    abstraction_prompts = [
        render_label_prompt(ex, label_template, few_shots, args.label_mode, args.max_solution_chars)
        for ex in dataset
    ]
    abstraction_messages = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=args.label_enable_thinking,
        )
        for prompt in abstraction_prompts
    ]

    abstraction_sampling = SamplingParams(
        temperature=1.0,
        top_p=0.95,
        top_k=20,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )

    print(f"Device name = {torch.cuda.get_device_name(0)}")
    llm_kwargs = {
        'model': args.base_model,
        'enable_lora': bool(args.abstraction_lora_path or args.solver_lora_path or args.judge_lora_path),
        'max_num_batched_tokens': args.max_num_batched_tokens,
        'gpu_memory_utilization': args.gpu_memory_utilization,
    }
    if args.max_model_len is not None:
        llm_kwargs['max_model_len'] = args.max_model_len
    llm = LLM(**llm_kwargs)
    abstraction_outputs = generate_with_optional_lora(
        llm,
        abstraction_messages,
        abstraction_sampling,
        args.abstraction_lora_path,
        "abstraction-generation",
    )

    solver_outputs = [None] * len(abstraction_outputs)
    predicted_answers = [None] * len(abstraction_outputs)
    solver_correct = [False] * len(abstraction_outputs)
    adherences = [None] * len(abstraction_outputs)
    abstractions = [extract_abstraction(text) for text in abstraction_outputs]

    if not args.skip_solver:
        solver_prompts = []
        solver_indices = []
        for idx, ex in enumerate(dataset):
            abstraction = abstractions[idx]
            if not abstraction:
                continue
            prompt = make_solver_prompt(ex["problem"], abstraction, solver_template)
            solver_prompts.append(
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=True,
                )
            )
            solver_indices.append(idx)

        solver_sampling = SamplingParams(
            temperature=args.temperature,
            top_p=0.95,
            top_k=20,
            max_tokens=args.max_tokens,
            seed=args.seed,
        )
        solver_generations = generate_with_optional_lora(
            llm,
            solver_prompts,
            solver_sampling,
            args.solver_lora_path,
            "solver",
        )

        for prompt_idx, dataset_idx in enumerate(solver_indices):
            text = solver_generations[prompt_idx]
            solver_outputs[dataset_idx] = text
            boxed = extract_answer_from_solution(text)
            predicted_answers[dataset_idx] = boxed
            expected = dataset[dataset_idx]["answer"]
            target = f"\boxed{{{expected}}}"
            if boxed is None:
                solver_correct[dataset_idx] = False
            else:
                try:
                    solver_correct[dataset_idx] = bool(verify(parse(target), parse(boxed)))
                except Exception:
                    solver_correct[dataset_idx] = False

        if args.compute_adherence and args.judge_lora_path:
            judge_prompts = []
            judge_indices = []
            for idx, ex in enumerate(dataset):
                if not abstractions[idx] or not solver_outputs[idx]:
                    continue
                judge_prompts.append(
                    tokenizer.apply_chat_template(
                        [{"role": "user", "content": make_judge_prompt(ex["problem"], abstractions[idx], solver_outputs[idx])}],
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=True,
                    )
                )
                judge_indices.append(idx)

            judge_sampling = SamplingParams(
                temperature=0.0,
                top_p=1.0,
                top_k=-1,
                max_tokens=256,
                seed=args.seed,
            )
            judge_outputs = generate_with_optional_lora(
                llm,
                judge_prompts,
                judge_sampling,
                args.judge_lora_path,
                "judge",
            )
            for prompt_idx, dataset_idx in enumerate(judge_indices):
                adherences[dataset_idx] = parse_judge_output(judge_outputs[prompt_idx])

    rows = []
    for idx, ex in enumerate(dataset):
        expected = ex["answer"]
        target = f"\boxed{{{expected}}}"
        if predicted_answers[idx] is None:
            recomputed_correct = False
        else:
            try:
                recomputed_correct = bool(verify(parse(target), parse(predicted_answers[idx])))
            except Exception:
                recomputed_correct = False
        rows.append(
            {
                "row_id": ex["row_id"],
                "problem": ex["problem"],
                "answer": ex["answer"],
                "baseline_passrate": ex["passrate"],
                "baseline_num_correct": ex["num_correct"],
                "selected_correct_indices": ex["selected_correct_indices"],
                "selected_incorrect_indices": ex["selected_incorrect_indices"],
                "label_mode": args.label_mode,
                "abstraction_prompt_text": abstraction_prompts[idx],
                "generated_abstraction_text": abstraction_outputs[idx],
                "abstraction": abstractions[idx],
                "solver_output_text": solver_outputs[idx],
                "solver_predicted_answer": predicted_answers[idx],
                "solver_correct": recomputed_correct,
                "adherence": adherences[idx],
            }
        )

    report = compute_accuracy(rows)
    report.update(
        {
            "title": args.report_title,
            "dataset_path": args.dataset_path,
            "split": args.split,
            "base_model": args.base_model,
            "label_mode": args.label_mode,
            "abstraction_prompt_path": args.abstraction_prompt_path,
            "solver_prompt_path": args.solver_prompt_path,
            "label_enable_thinking": args.label_enable_thinking,
            "few_shots_path": args.few_shots_path,
            "abstraction_lora_path": args.abstraction_lora_path,
            "solver_lora_path": args.solver_lora_path,
            "judge_lora_path": args.judge_lora_path,
            "max_model_len": args.max_model_len,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "gpu_memory_utilization": args.gpu_memory_utilization,
        }
    )

    (output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output_dir / "rows.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    summary_md = [
        f"# {args.report_title}",
        "",
        f"- dataset: `{args.dataset_path}` ({args.split})",
        f"- rows: {report['num_rows']}",
        f"- label mode: `{args.label_mode}`",
        f"- abstraction prompt: `{args.abstraction_prompt_path}`",
        f"- solver prompt: `{args.solver_prompt_path}`",
        f"- conditioned accuracy: {report['accuracy']:.4f}",
        f"- mean baseline passrate: {report['baseline_passrate_mean']:.4f}",
        f"- absolute accuracy lift: {report['absolute_accuracy_lift']:.4f}",
        f"- valid boxed answers: {report['num_valid_answers']}",
    ]
    if report['relative_accuracy_lift'] is not None:
        summary_md.append(f"- relative accuracy lift: {report['relative_accuracy_lift']:.4f}")
    if report['adherence_mean'] is not None:
        summary_md.append(f"- adherence mean: {report['adherence_mean']:.4f}")
        summary_md.append(f"- solve and adhere rate: {report['solve_and_adhere_rate']:.4f}")

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
