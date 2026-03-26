from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

import torch
from datasets import Dataset, DatasetDict, load_from_disk
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from contrastive_abstraction_utils import (
    classify_generated_answers,
    format_solution_block,
    read_text,
    repo_path,
    strip_think_blocks,
)


DEFAULT_BASE_MODEL = "Qwen/Qwen3-1.7B"
DEFAULT_PRINCIPLE_PROMPT = "prompt_templates/principle_extraction_template_v5.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dataset_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--output_dataset_path", type=str, default="")
    parser.add_argument("--split", type=str, default="")
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--base_model", type=str, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--principle_prompt_path", type=str, default=DEFAULT_PRINCIPLE_PROMPT)
    parser.add_argument("--label_temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--max_solution_chars", type=int, default=6000)
    parser.add_argument("--max_model_len", type=int, default=32768)
    parser.add_argument("--max_num_batched_tokens", type=int, default=4096)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--prompt_batch_size", type=int, default=64)
    parser.add_argument("--write_rows_json", action="store_true")
    return parser.parse_args()


def resolve_repo_file(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return repo_path(path_str)


def load_dataset_dict(path: str, split: str, max_rows: int | None) -> DatasetDict:
    loaded = load_from_disk(str(resolve_repo_file(path)))
    if isinstance(loaded, DatasetDict):
        if split:
            ds = loaded[split]
            if max_rows is not None:
                ds = ds.select(range(min(max_rows, ds.num_rows)))
            return DatasetDict({split: ds})
        built: dict[str, Dataset] = {}
        for name, ds in loaded.items():
            if max_rows is not None:
                ds = ds.select(range(min(max_rows, ds.num_rows)))
            built[name] = ds
        return DatasetDict(built)

    ds = loaded
    if max_rows is not None:
        ds = ds.select(range(min(max_rows, ds.num_rows)))
    return DatasetDict({split or "train": ds})


def ensure_trace_pairs(example: dict[str, Any]) -> tuple[list[tuple[int, str]], list[tuple[int, str]], list[bool]]:
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
    incorrect_indices = list(
        example.get("selected_incorrect_indices") or range(len(example.get("selected_incorrect_solutions") or []))
    )
    correct_solutions = list(example.get("selected_correct_solutions") or [])
    incorrect_solutions = list(example.get("selected_incorrect_solutions") or [])
    correct_pairs = list(zip(correct_indices, correct_solutions))
    incorrect_pairs = list(zip(incorrect_indices, incorrect_solutions))
    correctness = [True] * len(correct_pairs) + [False] * len(incorrect_pairs)
    return correct_pairs, incorrect_pairs, correctness


def cleaned_correct_traces(pairs: list[tuple[int, str]]) -> tuple[list[int], list[str]]:
    kept_indices: list[int] = []
    kept_traces: list[str] = []
    for trace_idx, trace_text in pairs:
        cleaned = strip_think_blocks(trace_text)
        if not cleaned:
            continue
        kept_indices.append(int(trace_idx))
        kept_traces.append(cleaned)
    return kept_indices, kept_traces


def render_principle_prompt(problem: str, correct_traces: list[str], template: str, max_solution_chars: int) -> str:
    correct_block = format_solution_block("Correct Trace", correct_traces, max_chars=max_solution_chars)
    return template.replace("{{PROBLEM}}", problem).replace("{{CORRECT_SOLUTIONS_BLOCK}}", correct_block)


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


def write_json(path: Path, payload: dict[str, Any] | list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    principle_lengths = [len((row.get("principles") or "").split()) for row in rows]
    trace_counts = [int(row.get("all_correct_trace_count", 0)) for row in rows]
    baseline_passrates = [float(row["baseline_passrate"]) for row in rows]
    baseline_num_correct = [float(row["baseline_num_correct"]) for row in rows]
    passrate_bucket_counts = Counter(str(float(row["baseline_passrate"])) for row in rows)
    trace_count_distribution = Counter(int(row.get("all_correct_trace_count", 0)) for row in rows)
    return {
        "num_rows": len(rows),
        "nonempty_principles": sum(1 for row in rows if row.get("principles")),
        "mean_principle_words": mean(principle_lengths) if principle_lengths else 0.0,
        "mean_correct_trace_count": mean(trace_counts) if trace_counts else 0.0,
        "max_correct_trace_count": max(trace_counts) if trace_counts else 0,
        "baseline_passrate_mean": mean(baseline_passrates) if baseline_passrates else 0.0,
        "baseline_num_correct_mean": mean(baseline_num_correct) if baseline_num_correct else 0.0,
        "passrate_bucket_counts": dict(sorted(passrate_bucket_counts.items())),
        "correct_trace_count_distribution": dict(sorted(trace_count_distribution.items())),
    }


def write_summary_md(path: Path, report: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    lines = [
        f"# {report['title']}",
        "",
        f"- input dataset: `{report['input_dataset_path']}`",
        f"- split: `{report['split']}`",
        f"- base model: `{report['base_model']}`",
        f"- principle prompt: `{report['principle_prompt_path']}`",
        f"- principle source: all non-empty correct traces",
        f"- kept rows: {report['num_rows']}",
        f"- non-empty principles: {report['nonempty_principles']}/{report['num_rows']}",
        f"- mean principle words: {report['mean_principle_words']:.2f}",
        f"- mean correct trace count: {report['mean_correct_trace_count']:.2f}",
        f"- max correct trace count: {report['max_correct_trace_count']}",
        f"- baseline passrate mean: {report['baseline_passrate_mean']:.4f}",
        f"- baseline num_correct mean: {report['baseline_num_correct_mean']:.4f}",
        "",
        "## Bucket Counts",
        "",
    ]

    for bucket, count in report["passrate_bucket_counts"].items():
        lines.append(f"- passrate {bucket}: {count}")

    lines.extend(["", "## Trace Count Distribution", ""])
    for trace_count, count in report["correct_trace_count_distribution"].items():
        lines.append(f"- {trace_count} correct traces: {count}")

    lines.extend(["", "## Sample Rows", ""])
    for row in rows[: min(5, len(rows))]:
        lines.extend(
            [
                f"### Row {row['row_id']}",
                row["problem"],
                "",
                f"- all correct trace indices: {row.get('all_correct_indices')}",
                f"- all correct trace count: {row.get('all_correct_trace_count')}",
                "",
                "#### Principles",
                row.get("principles") or "(empty)",
                "",
            ]
        )

    path.write_text("\n".join(lines), encoding="utf-8")


def build_rows_for_split(
    split_ds: Dataset,
    principle_template: str,
    tokenizer: AutoTokenizer,
    llm: LLM,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    base_rows: list[dict[str, Any]] = []
    prompts: list[str] = []

    for idx, ex in enumerate(split_ds):
        passrate = float(ex.get("passrate", 0.0))
        if passrate <= 0.0:
            continue

        correct_pairs, _, correctness = ensure_trace_pairs(ex)
        correct_indices, correct_traces = cleaned_correct_traces(correct_pairs)
        if not correct_traces:
            continue

        prompt = render_principle_prompt(
            ex["problem"],
            correct_traces,
            principle_template,
            max_solution_chars=args.max_solution_chars,
        )

        row_id = int(ex.get("row_id", idx))
        row = dict(ex)
        row["row_id"] = row_id
        row["solution_correctness"] = correctness
        row["baseline_num_correct"] = int(ex.get("num_correct", sum(correctness)))
        row["baseline_passrate"] = float(
            ex.get("passrate", (sum(correctness) / len(correctness)) if correctness else 0.0)
        )
        row["all_correct_indices"] = correct_indices
        row["all_correct_trace_count"] = len(correct_traces)
        row["all_correct_traces_text"] = correct_traces
        base_rows.append(row)
        prompts.append(prompt)

    sampling = SamplingParams(
        temperature=args.label_temperature,
        top_p=1.0 if args.label_temperature == 0.0 else 0.95,
        top_k=-1 if args.label_temperature == 0.0 else 20,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )
    outputs = generate_texts(llm, tokenizer, prompts, sampling, args.prompt_batch_size)

    rows: list[dict[str, Any]] = []
    for row, principle_text in zip(base_rows, outputs):
        row["generated_principles_text"] = principle_text
        row["principles"] = principle_text.strip()
        rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    dataset_dict = load_dataset_dict(args.input_dataset_path, args.split, args.max_rows)
    principle_template = read_text(resolve_repo_file(args.principle_prompt_path))

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
    saved_rows_by_split: dict[str, list[dict[str, Any]]] = {}
    split_reports: dict[str, dict[str, Any]] = {}
    for split_name, split_ds in dataset_dict.items():
        print(f"Processing split: {split_name} rows={split_ds.num_rows}")
        rows = build_rows_for_split(split_ds, principle_template, tokenizer, llm, args)
        saved_rows_by_split[split_name] = rows
        built_splits[split_name] = Dataset.from_list(rows)
        split_reports[split_name] = summarize_rows(rows)
        print(f"Built labels for split {split_name}: {len(rows)} rows")

    if args.output_dataset_path:
        output_dataset_path = resolve_repo_file(args.output_dataset_path)
        output_dataset_path.parent.mkdir(parents=True, exist_ok=True)
        if len(built_splits) == 1 and not args.split:
            next(iter(built_splits.values())).save_to_disk(str(output_dataset_path))
        else:
            DatasetDict(built_splits).save_to_disk(str(output_dataset_path))
        print(f"Saved labeled dataset to {output_dataset_path}")

    if args.output_dir:
        output_dir = resolve_repo_file(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        split_name = args.split or next(iter(saved_rows_by_split))
        rows = saved_rows_by_split[split_name]
        report = dict(split_reports[split_name])
        report.update(
            {
                "title": "Principle labels from all correct traces",
                "input_dataset_path": args.input_dataset_path,
                "split": split_name,
                "base_model": args.base_model,
                "principle_prompt_path": args.principle_prompt_path,
                "max_model_len": args.max_model_len,
                "max_num_batched_tokens": args.max_num_batched_tokens,
                "gpu_memory_utilization": args.gpu_memory_utilization,
            }
        )
        write_json(output_dir / "report.json", report)
        if args.write_rows_json:
            write_json(output_dir / "rows.json", rows)
        write_json(output_dir / "sample_rows.json", rows[: min(50, len(rows))])
        write_summary_md(output_dir / "summary.md", report, rows)
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
