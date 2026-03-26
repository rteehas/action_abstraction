from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Optional

import torch
from datasets import Dataset, DatasetDict, load_from_disk
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from contrastive_abstraction_utils import (
    classify_generated_answers,
    extract_abstraction,
    format_solution_block,
    read_text,
    repo_path,
    strip_think_blocks,
)
from process_deepscaler_dataset import extract_answer_from_solution, parse, verify


DEFAULT_BASE_MODEL = "Qwen/Qwen3-1.7B"
DEFAULT_PRINCIPLE_PROMPT = "prompt_templates/principle_extraction_template_v2.txt"
DEFAULT_TEACHER_PROMPT = "prompt_templates/contrastive_abstraction_labeling_prompt_zero_shot_v16.txt"
DEFAULT_SOLVER_PROMPT = "prompt_templates/hint_conditioned_problem_solving_rich_v1.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dataset_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--output_dataset_path", type=str, default="")
    parser.add_argument("--split", type=str, default="")
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--base_model", type=str, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--principle_prompt_path", type=str, default=DEFAULT_PRINCIPLE_PROMPT)
    parser.add_argument("--teacher_prompt_path", type=str, default=DEFAULT_TEACHER_PROMPT)
    parser.add_argument("--teacher_few_shots_path", type=str, default="")
    parser.add_argument("--solver_prompt_path", type=str, default=DEFAULT_SOLVER_PROMPT)
    parser.add_argument("--label_temperature", type=float, default=1.0)
    parser.add_argument("--solver_temperature", type=float, default=0.6)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--max_solution_chars", type=int, default=6000)
    parser.add_argument("--max_model_len", type=int, default=32768)
    parser.add_argument("--max_num_batched_tokens", type=int, default=4096)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--prompt_batch_size", type=int, default=64)
    parser.add_argument("--skip_solver", action="store_true")
    parser.add_argument("--filter_empty", action="store_true")
    return parser.parse_args()


def resolve_repo_file(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return repo_path(path_str)


def maybe_limit(ds: Dataset, max_rows: Optional[int]) -> Dataset:
    if max_rows is None:
        return ds
    return ds.select(range(min(max_rows, ds.num_rows)))


def load_dataset_dict(path: str, split: str, max_rows: Optional[int], seed: int, val_frac: float) -> DatasetDict:
    loaded = load_from_disk(str(resolve_repo_file(path)))
    if isinstance(loaded, DatasetDict):
        if split:
            return DatasetDict({split: maybe_limit(loaded[split], max_rows)})
        return DatasetDict({name: maybe_limit(ds, max_rows) for name, ds in loaded.items()})

    loaded = maybe_limit(loaded, max_rows)
    if split:
        return DatasetDict({split: loaded})
    if loaded.num_rows <= 1:
        return DatasetDict({"train": loaded, "test": loaded})
    return loaded.train_test_split(test_size=val_frac, seed=seed)


def ensure_trace_pairs(example: dict) -> tuple[list[tuple[int, str]], list[tuple[int, str]], list[bool]]:
    generated_solutions = example.get("generated_solution")
    generated_answers = example.get("generated_answer")
    if generated_solutions is not None and generated_answers is not None:
        solutions = list(generated_solutions)
        answers = list(generated_answers)
        if example.get("solution_correctness") is not None:
            correctness = [bool(value) for value in example["solution_correctness"]]
        else:
            correctness = classify_generated_answers(example["answer"], answers)
        correct_pairs = [(idx, solutions[idx]) for idx, is_correct in enumerate(correctness) if is_correct]
        incorrect_pairs = [(idx, solutions[idx]) for idx, is_correct in enumerate(correctness) if not is_correct]
        return correct_pairs, incorrect_pairs, correctness

    correct_indices = list(example.get("selected_correct_indices") or range(len(example.get("selected_correct_solutions") or [])))
    incorrect_indices = list(example.get("selected_incorrect_indices") or range(len(example.get("selected_incorrect_solutions") or [])))
    correct_solutions = list(example.get("selected_correct_solutions") or [])
    incorrect_solutions = list(example.get("selected_incorrect_solutions") or [])
    correct_pairs = list(zip(correct_indices, correct_solutions))
    incorrect_pairs = list(zip(incorrect_indices, incorrect_solutions))
    correctness = [True] * len(correct_pairs) + [False] * len(incorrect_pairs)
    return correct_pairs, incorrect_pairs, correctness


def first_nonempty_trace(pairs: list[tuple[int, str]]) -> tuple[Optional[int], str]:
    for trace_idx, trace_text in pairs:
        if strip_think_blocks(trace_text):
            return trace_idx, trace_text
    return None, ""


def render_principle_prompt(problem: str, first_correct_trace: str, template: str, max_solution_chars: int) -> str:
    correct_block = format_solution_block("Correct Trace", [first_correct_trace], max_chars=max_solution_chars)
    return template.replace("{{PROBLEM}}", problem).replace("{{CORRECT_SOLUTIONS_BLOCK}}", correct_block)


def render_teacher_prompt(
    problem: str,
    correct_traces: list[str],
    incorrect_traces: list[str],
    template: str,
    few_shots: str,
    max_solution_chars: int,
) -> str:
    correct_block = format_solution_block("Correct Trace", correct_traces, max_chars=max_solution_chars)
    incorrect_block = format_solution_block("Incorrect Trace", incorrect_traces, max_chars=max_solution_chars)
    return (
        template.replace("{{FEW_SHOT_EXAMPLES}}", few_shots)
        .replace("{{PROBLEM}}", problem)
        .replace("{{CORRECT_SOLUTIONS_BLOCK}}", correct_block)
        .replace("{{INCORRECT_SOLUTIONS_BLOCK}}", incorrect_block)
    )


def combine_abstraction(principles_text: str, teacher_note: str) -> str:
    parts: list[str] = []
    clean_teacher = (teacher_note or "").strip()
    clean_principles = (principles_text or "").strip()
    if clean_principles:
        parts.append(f"Principles:\n{clean_principles}")
    if clean_teacher:
        parts.append(f"Teacher Note:\n{clean_teacher}")
    return "\n\n".join(parts).strip()


def apply_chat_template(tokenizer: AutoTokenizer, prompts: list[str]) -> list[str]:
    return [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for prompt in prompts
    ]


def generate_texts(
    llm: LLM,
    tokenizer: AutoTokenizer,
    prompts: list[str],
    sampling: SamplingParams,
    batch_size: int,
) -> list[str]:
    outputs: list[str] = []
    total = len(prompts)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        message_batch = apply_chat_template(tokenizer, prompts[start:end])
        batch_outputs = llm.generate(message_batch, sampling)
        outputs.extend(output.outputs[0].text for output in batch_outputs)
        print(f"Generated {end}/{total} prompts")
    return outputs


def compute_accuracy(rows: list[dict]) -> dict:
    total = len(rows)
    correct = sum(1 for row in rows if row.get("solver_correct"))
    valid = sum(1 for row in rows if row.get("solver_predicted_answer") is not None)
    baseline_passrate_mean = mean(row["baseline_passrate"] for row in rows) if rows else 0.0
    baseline_num_correct_mean = mean(row["baseline_num_correct"] for row in rows) if rows else 0.0
    accuracy = correct / total if total else 0.0
    return {
        "num_rows": total,
        "num_correct": correct,
        "accuracy": accuracy,
        "num_valid_answers": valid,
        "baseline_passrate_mean": baseline_passrate_mean,
        "baseline_num_correct_mean": baseline_num_correct_mean,
        "absolute_accuracy_lift": accuracy - baseline_passrate_mean,
        "relative_accuracy_lift": ((accuracy - baseline_passrate_mean) / baseline_passrate_mean) if baseline_passrate_mean else None,
    }


def summarize_label_stats(rows: list[dict]) -> dict:
    principle_lengths = [len((row["principles"] or "").split()) for row in rows]
    teacher_lengths = [len((row["teacher_note"] or "").split()) for row in rows]
    combined_lengths = [len((row["abstraction"] or "").split()) for row in rows]
    return {
        "num_rows": len(rows),
        "nonempty_principles": sum(1 for row in rows if row.get("principles")),
        "nonempty_teacher_notes": sum(1 for row in rows if row.get("teacher_note")),
        "nonempty_abstractions": sum(1 for row in rows if row.get("abstraction")),
        "mean_principle_words": mean(principle_lengths) if principle_lengths else 0.0,
        "mean_teacher_note_words": mean(teacher_lengths) if teacher_lengths else 0.0,
        "mean_combined_words": mean(combined_lengths) if combined_lengths else 0.0,
    }


def score_solver_outputs(rows: list[dict], solver_outputs: list[Optional[str]]) -> None:
    for row, solver_text in zip(rows, solver_outputs):
        row["solver_output_text"] = solver_text
        boxed = extract_answer_from_solution(solver_text) if solver_text else None
        row["solver_predicted_answer"] = boxed
        target = f"\\boxed{{{row['answer']}}}"
        if boxed is None:
            row["solver_correct"] = False
            continue
        try:
            row["solver_correct"] = bool(verify(parse(target), parse(boxed)))
        except Exception:
            row["solver_correct"] = False


def run_solver(
    llm: LLM,
    tokenizer: AutoTokenizer,
    rows: list[dict],
    solver_template: str,
    temperature: float,
    max_tokens: int,
    seed: int,
    batch_size: int,
) -> None:
    prompts: list[str] = []
    prompt_row_indices: list[int] = []
    for idx, row in enumerate(rows):
        abstraction = row.get("abstraction") or ""
        if not abstraction:
            continue
        prompts.append(solver_template.replace("{{PROBLEM}}", row["problem"]).replace("{{ABSTRACTION}}", abstraction))
        prompt_row_indices.append(idx)

    if not prompts:
        score_solver_outputs(rows, [None] * len(rows))
        return

    sampling = SamplingParams(
        temperature=temperature,
        top_p=0.95,
        top_k=20,
        max_tokens=max_tokens,
        seed=seed,
    )
    solver_outputs_by_index = [None] * len(rows)
    generated = generate_texts(llm, tokenizer, prompts, sampling, batch_size=batch_size)
    for row_idx, output_text in zip(prompt_row_indices, generated):
        solver_outputs_by_index[row_idx] = output_text
    score_solver_outputs(rows, solver_outputs_by_index)


def write_json(path: Path, payload: dict | list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_summary_md(path: Path, report: dict, rows: list[dict]) -> None:
    lines = [
        f"# {report['title']}",
        "",
        f"- input dataset: `{report['input_dataset_path']}`",
        f"- split: `{report['split']}`",
        f"- base model: `{report['base_model']}`",
        f"- principle prompt: `{report['principle_prompt_path']}`",
        f"- teacher prompt: `{report['teacher_prompt_path']}`",
        f"- principle source: first non-empty correct trace only",
        f"- teacher-note source: all correct and incorrect traces",
        f"- non-empty abstractions: {report['nonempty_abstractions']}/{report['num_rows']}",
        f"- mean principle words: {report['mean_principle_words']:.2f}",
        f"- mean teacher-note words: {report['mean_teacher_note_words']:.2f}",
        f"- mean combined words: {report['mean_combined_words']:.2f}",
    ]
    if "accuracy" in report:
        lines.extend(
            [
                f"- accuracy: {report['num_correct']}/{report['num_rows']} = {report['accuracy']:.4f}",
                f"- baseline passrate mean: {report['baseline_passrate_mean']:.4f}",
                f"- absolute lift: {report['absolute_accuracy_lift']:.4f}",
                f"- valid boxed answers: {report['num_valid_answers']}",
            ]
        )

    preview_rows = rows[: min(5, len(rows))]
    if preview_rows:
        lines.extend(["", "## Sample Rows", ""])
    for row in preview_rows:
        lines.extend(
            [
                f"### Row {row['row_id']}",
                row["problem"],
                "",
                "#### Principles",
                row.get("principles") or "(empty)",
                "",
                "#### Teacher Note",
                row.get("teacher_note") or "(empty)",
                "",
                "#### Combined Abstraction",
                row.get("abstraction") or "(empty)",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def build_rows_for_split(
    split_ds: Dataset,
    principle_template: str,
    teacher_template: str,
    teacher_few_shots: str,
    tokenizer: AutoTokenizer,
    llm: LLM,
    args: argparse.Namespace,
) -> list[dict]:
    base_rows: list[dict] = []
    principle_prompts: list[str] = []
    teacher_prompts: list[str] = []

    for idx, ex in enumerate(split_ds):
        correct_pairs, incorrect_pairs, correctness = ensure_trace_pairs(ex)
        first_correct_idx, first_correct_trace = first_nonempty_trace(correct_pairs)
        all_correct_traces = [trace for _, trace in correct_pairs]
        all_incorrect_traces = [trace for _, trace in incorrect_pairs]

        principle_prompt = render_principle_prompt(
            ex["problem"],
            first_correct_trace,
            principle_template,
            max_solution_chars=args.max_solution_chars,
        )
        teacher_prompt = render_teacher_prompt(
            ex["problem"],
            all_correct_traces,
            all_incorrect_traces,
            teacher_template,
            teacher_few_shots,
            max_solution_chars=args.max_solution_chars,
        )

        base_row = dict(ex)
        base_row["row_id"] = ex.get("row_id", idx)
        base_row["solution_correctness"] = correctness
        base_row["baseline_num_correct"] = int(ex.get("num_correct", sum(correctness)))
        base_row["baseline_passrate"] = float(
            ex.get("passrate", (sum(correctness) / len(correctness)) if correctness else 0.0)
        )
        base_row["all_correct_indices"] = [trace_idx for trace_idx, _ in correct_pairs]
        base_row["all_incorrect_indices"] = [trace_idx for trace_idx, _ in incorrect_pairs]
        base_row["first_correct_trace_index"] = first_correct_idx
        base_row["first_correct_trace_text"] = strip_think_blocks(first_correct_trace)
        base_row["principle_prompt_text"] = principle_prompt
        base_row["teacher_prompt_text"] = teacher_prompt
        base_rows.append(base_row)
        principle_prompts.append(principle_prompt)
        teacher_prompts.append(teacher_prompt)

    sampling = SamplingParams(
        temperature=args.label_temperature,
        top_p=0.95,
        top_k=20,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )
    principle_outputs = generate_texts(llm, tokenizer, principle_prompts, sampling, args.prompt_batch_size)
    teacher_outputs = generate_texts(llm, tokenizer, teacher_prompts, sampling, args.prompt_batch_size)

    rows: list[dict] = []
    for base_row, principle_text, teacher_text in zip(base_rows, principle_outputs, teacher_outputs):
        teacher_note = extract_abstraction(teacher_text) or ""
        combined = combine_abstraction(principle_text, teacher_note)
        base_row["generated_principles_text"] = principle_text
        base_row["principles"] = principle_text.strip()
        base_row["generated_teacher_note_text"] = teacher_text
        base_row["teacher_note"] = teacher_note
        base_row["abstraction"] = combined
        rows.append(base_row)

    if args.filter_empty:
        rows = [row for row in rows if row.get("abstraction")]
    return rows


def main() -> None:
    args = parse_args()
    dataset_dict = load_dataset_dict(args.input_dataset_path, args.split, args.max_rows, args.seed, args.val_frac)

    principle_template = read_text(resolve_repo_file(args.principle_prompt_path))
    teacher_template = read_text(resolve_repo_file(args.teacher_prompt_path))
    teacher_few_shots = read_text(resolve_repo_file(args.teacher_few_shots_path)) if args.teacher_few_shots_path else ""
    solver_template = read_text(resolve_repo_file(args.solver_prompt_path))

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

    built_splits: dict[str, Dataset] = {}
    saved_rows_by_split: dict[str, list[dict]] = {}
    for split_name, split_ds in dataset_dict.items():
        print(f"Processing split: {split_name} rows={split_ds.num_rows}")
        rows = build_rows_for_split(split_ds, principle_template, teacher_template, teacher_few_shots, tokenizer, llm, args)
        saved_rows_by_split[split_name] = rows
        built_splits[split_name] = Dataset.from_list(rows)
        print(f"Built labels for split {split_name}: {len(rows)} rows")

    if args.output_dataset_path:
        output_dataset_path = resolve_repo_file(args.output_dataset_path)
        output_dataset_path.parent.mkdir(parents=True, exist_ok=True)
        DatasetDict(built_splits).save_to_disk(str(output_dataset_path))
        print(f"Saved labeled dataset to {output_dataset_path}")

    if args.output_dir:
        output_dir = resolve_repo_file(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        split_name = args.split or ("test" if "test" in saved_rows_by_split else next(iter(saved_rows_by_split)))
        rows = saved_rows_by_split[split_name]

        if not args.skip_solver:
            run_solver(
                llm,
                tokenizer,
                rows,
                solver_template,
                temperature=args.solver_temperature,
                max_tokens=args.max_tokens,
                seed=args.seed,
                batch_size=args.prompt_batch_size,
            )
        else:
            for row in rows:
                row["solver_output_text"] = None
                row["solver_predicted_answer"] = None
                row["solver_correct"] = False

        report = summarize_label_stats(rows)
        if not args.skip_solver:
            report.update(compute_accuracy(rows))
        report.update(
            {
                "title": "Combined principle v2 + v16 teacher-note labels",
                "input_dataset_path": args.input_dataset_path,
                "split": split_name,
                "base_model": args.base_model,
                "principle_prompt_path": args.principle_prompt_path,
                "teacher_prompt_path": args.teacher_prompt_path,
                "teacher_few_shots_path": args.teacher_few_shots_path,
                "solver_prompt_path": args.solver_prompt_path,
                "max_model_len": args.max_model_len,
                "max_num_batched_tokens": args.max_num_batched_tokens,
                "gpu_memory_utilization": args.gpu_memory_utilization,
            }
        )
        write_json(output_dir / "rows.json", rows)
        write_json(output_dir / "report.json", report)
        write_summary_md(output_dir / "summary.md", report, rows)
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
