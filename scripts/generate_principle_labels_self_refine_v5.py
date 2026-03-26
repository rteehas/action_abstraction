from __future__ import annotations

import argparse
import json
import re
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
DEFAULT_INITIAL_PROMPT = "prompt_templates/principle_extraction_template_v5.txt"
DEFAULT_AUDITOR_PROMPT = "prompt_templates/principle_extraction_auditor.txt"
DEFAULT_REVISOR_PROMPT = "prompt_templates/principle_extraction_revisor.txt"
DEFAULT_CHOOSER_PROMPT = "prompt_templates/principle_extraction_chooser.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dataset_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--output_dataset_path", type=str, default="")
    parser.add_argument("--split", type=str, default="")
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--row_ids", type=str, default="")
    parser.add_argument("--base_model", type=str, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--principle_prompt_path", type=str, default=DEFAULT_INITIAL_PROMPT)
    parser.add_argument("--auditor_prompt_path", type=str, default=DEFAULT_AUDITOR_PROMPT)
    parser.add_argument("--revisor_prompt_path", type=str, default=DEFAULT_REVISOR_PROMPT)
    parser.add_argument("--chooser_prompt_path", type=str, default=DEFAULT_CHOOSER_PROMPT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_solution_chars", type=int, default=6000)
    parser.add_argument("--max_model_len", type=int, default=32768)
    parser.add_argument("--max_num_batched_tokens", type=int, default=4096)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--prompt_batch_size", type=int, default=64)
    parser.add_argument("--initial_temperature", type=float, default=0.0)
    parser.add_argument("--audit_temperature", type=float, default=0.0)
    parser.add_argument("--revise_temperature", type=float, default=0.0)
    parser.add_argument("--choose_temperature", type=float, default=0.0)
    parser.add_argument("--initial_max_tokens", type=int, default=4096)
    parser.add_argument("--audit_max_tokens", type=int, default=1024)
    parser.add_argument("--revise_max_tokens", type=int, default=4096)
    parser.add_argument("--choose_max_tokens", type=int, default=512)
    parser.add_argument("--write_rows_json", action="store_true")
    parser.add_argument(
        "--no_reuse_existing_principles",
        action="store_true",
        help="Force regeneration with the v5 extraction prompt even when generated_principles_text already exists.",
    )
    return parser.parse_args()


def resolve_repo_file(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return repo_path(path_str)


def load_dataset_dict(path: str, split: str) -> DatasetDict:
    loaded = load_from_disk(str(resolve_repo_file(path)))
    if isinstance(loaded, DatasetDict):
        if split:
            return DatasetDict({split: loaded[split]})
        return loaded
    return DatasetDict({split or "train": loaded})


def parse_row_ids(row_ids_arg: str) -> set[int]:
    values: set[int] = set()
    for raw in row_ids_arg.split(","):
        token = raw.strip()
        if not token:
            continue
        values.add(int(token))
    return values


def subset_splits(dataset_dict: DatasetDict, row_ids: set[int], max_rows: int | None) -> DatasetDict:
    built: dict[str, Dataset] = {}
    for split_name, ds in dataset_dict.items():
        if row_ids:
            keep_indices = [idx for idx, row in enumerate(ds) if int(row.get("row_id", idx)) in row_ids]
            ds = ds.select(keep_indices)
        if max_rows is not None:
            ds = ds.select(range(min(max_rows, ds.num_rows)))
        built[split_name] = ds
    return DatasetDict(built)


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


def render_refinement_prompt(
    template: str,
    problem: str,
    correct_traces: list[str],
    max_solution_chars: int,
    principles_text: str = "",
    critic_report: str = "",
    version_a: str = "",
    version_b: str = "",
) -> str:
    correct_block = format_solution_block("Correct Trace", correct_traces, max_chars=max_solution_chars)
    replacements = {
        "{{PROBLEM}}": problem,
        "{{CORRECT_SOLUTIONS_BLOCK}}": correct_block,
        "{{PRINCIPLES_BLOCK}}": principles_text.strip() or "(empty)",
        "{{CRITIC_REPORT}}": critic_report.strip() or "{}",
        "{{VERSION_A}}": version_a.strip() or "(empty)",
        "{{VERSION_B}}": version_b.strip() or "(empty)",
    }
    rendered = template
    for placeholder, value in replacements.items():
        rendered = rendered.replace(placeholder, value)
    return rendered


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


def build_sampling_params(temperature: float, max_tokens: int, seed: int) -> SamplingParams:
    return SamplingParams(
        temperature=temperature,
        top_p=1.0 if temperature == 0.0 else 0.95,
        top_k=-1 if temperature == 0.0 else 20,
        max_tokens=max_tokens,
        seed=seed,
    )


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


def extract_first_json_object(text: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    for idx, char in enumerate(text):
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(text[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def normalize_verdict(report: dict[str, Any] | None) -> str:
    if not report:
        return "keep"
    if not bool(report.get("main_lever_present", True)):
        return "revise"
    issue_keys = (
        "duplicates",
        "procedural_not_principles",
        "unsupported_claims",
        "missing_from_trace",
        "priority_edits",
    )
    if any(report.get(key) for key in issue_keys):
        return "revise"
    verdict = str(report.get("verdict", "")).strip().lower()
    return "revise" if verdict == "revise" else "keep"


def normalize_winner(report: dict[str, Any] | None) -> str:
    if not report:
        return "A"
    winner = str(report.get("winner", "")).strip().upper()
    return "B" if winner == "B" else "A"


def extract_winner_from_text(text: str) -> str | None:
    match = re.search(r'"winner"\s*:\s*"([AB])"', text)
    if match:
        return match.group(1)
    match = re.search(r'\bwinner\b[^AB]*([AB])\b', text, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return None


def write_json(path: Path, payload: dict[str, Any] | list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    principle_lengths = [len((row.get("generated_principles_text") or "").split()) for row in rows]
    trace_counts = [int(row.get("all_correct_trace_count", 0)) for row in rows]
    baseline_passrates = [float(row["baseline_passrate"]) for row in rows]
    passrate_bucket_counts = Counter(str(float(row["baseline_passrate"])) for row in rows)
    trace_count_distribution = Counter(int(row.get("all_correct_trace_count", 0)) for row in rows)
    auditor_verdict_counts = Counter(str(row.get("auditor_verdict", "keep")) for row in rows)
    chooser_counts = Counter(str(row.get("chooser_winner", "SKIP")) for row in rows)
    final_source_counts = Counter(str(row.get("final_principles_source", "initial")) for row in rows)
    initial_source_counts = Counter(str(row.get("initial_principles_source", "generated")) for row in rows)
    return {
        "num_rows": len(rows),
        "nonempty_principles": sum(1 for row in rows if row.get("generated_principles_text")),
        "mean_principle_words": mean(principle_lengths) if principle_lengths else 0.0,
        "mean_correct_trace_count": mean(trace_counts) if trace_counts else 0.0,
        "max_correct_trace_count": max(trace_counts) if trace_counts else 0,
        "baseline_passrate_mean": mean(baseline_passrates) if baseline_passrates else 0.0,
        "passrate_bucket_counts": dict(sorted(passrate_bucket_counts.items())),
        "correct_trace_count_distribution": dict(sorted(trace_count_distribution.items())),
        "auditor_verdict_counts": dict(sorted(auditor_verdict_counts.items())),
        "chooser_counts": dict(sorted(chooser_counts.items())),
        "final_source_counts": dict(sorted(final_source_counts.items())),
        "initial_source_counts": dict(sorted(initial_source_counts.items())),
        "num_changed_after_refinement": sum(
            1
            for row in rows
            if (row.get("initial_generated_principles_text") or "").strip()
            != (row.get("generated_principles_text") or "").strip()
        ),
    }


def write_summary_md(path: Path, report: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    lines = [
        f"# {report['title']}",
        "",
        f"- input dataset: `{report['input_dataset_path']}`",
        f"- split: `{report['split']}`",
        f"- base model: `{report['base_model']}`",
        f"- initial prompt: `{report['principle_prompt_path']}`",
        f"- auditor prompt: `{report['auditor_prompt_path']}`",
        f"- revisor prompt: `{report['revisor_prompt_path']}`",
        f"- chooser prompt: `{report['chooser_prompt_path']}`",
        f"- kept rows: {report['num_rows']}",
        f"- non-empty final principles: {report['nonempty_principles']}/{report['num_rows']}",
        f"- mean final principle words: {report['mean_principle_words']:.2f}",
        f"- rows changed after refinement: {report['num_changed_after_refinement']}",
        "",
        "## Verdict Counts",
        "",
    ]

    for verdict, count in report["auditor_verdict_counts"].items():
        lines.append(f"- auditor {verdict}: {count}")
    for winner, count in report["chooser_counts"].items():
        lines.append(f"- chooser {winner}: {count}")
    for source, count in report["final_source_counts"].items():
        lines.append(f"- final source {source}: {count}")

    lines.extend(["", "## Sample Rows", ""])
    for row in rows[: min(3, len(rows))]:
        lines.extend(
            [
                f"### Row {row['row_id']}",
                row["problem"],
                "",
                f"- auditor verdict: {row.get('auditor_verdict')}",
                f"- chooser winner: {row.get('chooser_winner')}",
                f"- final source: {row.get('final_principles_source')}",
                "",
                "#### Initial Principles",
                row.get("initial_generated_principles_text") or "(empty)",
                "",
                "#### Final Principles",
                row.get("generated_principles_text") or "(empty)",
                "",
            ]
        )

    path.write_text("\n".join(lines), encoding="utf-8")


def prepare_rows(split_ds: Dataset, reuse_existing_principles: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, ex in enumerate(split_ds):
        passrate = float(ex.get("baseline_passrate", ex.get("passrate", 0.0)))
        if passrate <= 0.0:
            continue

        correct_pairs, _, correctness = ensure_trace_pairs(ex)
        correct_indices, correct_traces = cleaned_correct_traces(correct_pairs)
        if not correct_traces:
            continue

        row_id = int(ex.get("row_id", idx))
        row = dict(ex)
        row["row_id"] = row_id
        row["solution_correctness"] = correctness
        row["baseline_num_correct"] = int(ex.get("baseline_num_correct", ex.get("num_correct", sum(correctness))))
        row["baseline_passrate"] = float(ex.get("baseline_passrate", passrate))
        row["all_correct_indices"] = correct_indices
        row["all_correct_trace_count"] = len(correct_traces)
        row["all_correct_traces_text"] = correct_traces

        existing = str(ex.get("generated_principles_text") or ex.get("principles") or "").strip()
        if reuse_existing_principles and existing:
            row["initial_generated_principles_text"] = existing
            row["initial_principles_source"] = "existing"
        else:
            row["initial_generated_principles_text"] = ""
            row["initial_principles_source"] = "generated"
        rows.append(row)
    return rows


def build_initial_principles(
    rows: list[dict[str, Any]],
    template: str,
    llm: LLM,
    tokenizer: AutoTokenizer,
    args: argparse.Namespace,
) -> None:
    prompts: list[str] = []
    pending_indices: list[int] = []
    for idx, row in enumerate(rows):
        if row.get("initial_generated_principles_text"):
            continue
        prompts.append(
            render_principle_prompt(
                row["problem"],
                list(row["all_correct_traces_text"]),
                template,
                max_solution_chars=args.max_solution_chars,
            )
        )
        pending_indices.append(idx)
    if not prompts:
        return

    outputs = generate_texts(
        llm,
        tokenizer,
        prompts,
        build_sampling_params(args.initial_temperature, args.initial_max_tokens, args.seed),
        args.prompt_batch_size,
    )
    for row_idx, output in zip(pending_indices, outputs):
        rows[row_idx]["initial_generated_principles_text"] = output.strip()


def run_auditor(
    rows: list[dict[str, Any]],
    template: str,
    llm: LLM,
    tokenizer: AutoTokenizer,
    args: argparse.Namespace,
) -> None:
    prompts = [
        render_refinement_prompt(
            template,
            row["problem"],
            list(row["all_correct_traces_text"]),
            args.max_solution_chars,
            principles_text=row["initial_generated_principles_text"],
        )
        for row in rows
    ]
    outputs = generate_texts(
        llm,
        tokenizer,
        prompts,
        build_sampling_params(args.audit_temperature, args.audit_max_tokens, args.seed),
        args.prompt_batch_size,
    )
    for row, output in zip(rows, outputs):
        report = extract_first_json_object(output)
        row["auditor_raw_output"] = output
        row["auditor_report_json"] = json.dumps(report, indent=2) if report else ""
        row["auditor_verdict"] = normalize_verdict(report)


def run_revisor(
    rows: list[dict[str, Any]],
    template: str,
    llm: LLM,
    tokenizer: AutoTokenizer,
    args: argparse.Namespace,
) -> None:
    prompts: list[str] = []
    pending_indices: list[int] = []
    for idx, row in enumerate(rows):
        row["revised_generated_principles_text"] = ""
        if row.get("auditor_verdict") != "revise":
            continue
        prompts.append(
            render_refinement_prompt(
                template,
                row["problem"],
                list(row["all_correct_traces_text"]),
                args.max_solution_chars,
                principles_text=row["initial_generated_principles_text"],
                critic_report=row.get("auditor_report_json") or row.get("auditor_raw_output") or "{}",
            )
        )
        pending_indices.append(idx)
    if not prompts:
        return

    outputs = generate_texts(
        llm,
        tokenizer,
        prompts,
        build_sampling_params(args.revise_temperature, args.revise_max_tokens, args.seed),
        args.prompt_batch_size,
    )
    for row_idx, output in zip(pending_indices, outputs):
        rows[row_idx]["revised_generated_principles_text"] = output.strip()


def run_chooser(
    rows: list[dict[str, Any]],
    template: str,
    llm: LLM,
    tokenizer: AutoTokenizer,
    args: argparse.Namespace,
) -> None:
    prompts: list[str] = []
    pending_indices: list[int] = []
    for idx, row in enumerate(rows):
        row["chooser_raw_output"] = ""
        row["chooser_decision_json"] = ""
        row["chooser_winner"] = "SKIP"
        revised = (row.get("revised_generated_principles_text") or "").strip()
        initial = (row.get("initial_generated_principles_text") or "").strip()
        if row.get("auditor_verdict") != "revise" or not revised or revised == initial:
            continue
        prompts.append(
            render_refinement_prompt(
                template,
                row["problem"],
                list(row["all_correct_traces_text"]),
                args.max_solution_chars,
                version_a=initial,
                version_b=revised,
            )
        )
        pending_indices.append(idx)
    if prompts:
        outputs = generate_texts(
            llm,
            tokenizer,
            prompts,
            build_sampling_params(args.choose_temperature, args.choose_max_tokens, args.seed),
            args.prompt_batch_size,
        )
        for row_idx, output in zip(pending_indices, outputs):
            decision = extract_first_json_object(output)
            rows[row_idx]["chooser_raw_output"] = output
            rows[row_idx]["chooser_decision_json"] = json.dumps(decision, indent=2) if decision else ""
            if decision is not None:
                rows[row_idx]["chooser_winner"] = normalize_winner(decision)
            else:
                rows[row_idx]["chooser_winner"] = extract_winner_from_text(output) or "A"

    for row in rows:
        initial = (row.get("initial_generated_principles_text") or "").strip()
        revised = (row.get("revised_generated_principles_text") or "").strip()
        winner = row.get("chooser_winner")
        if winner == "B" and revised:
            final = revised
            source = "revised"
        else:
            final = initial
            source = "initial"
        row["final_principles_source"] = source
        row["generated_principles_text"] = final
        row["principles"] = final.strip()


def main() -> None:
    args = parse_args()
    dataset_dict = load_dataset_dict(args.input_dataset_path, args.split)
    dataset_dict = subset_splits(dataset_dict, parse_row_ids(args.row_ids), args.max_rows)

    principle_template = read_text(resolve_repo_file(args.principle_prompt_path))
    auditor_template = read_text(resolve_repo_file(args.auditor_prompt_path))
    revisor_template = read_text(resolve_repo_file(args.revisor_prompt_path))
    chooser_template = read_text(resolve_repo_file(args.chooser_prompt_path))

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
    reuse_existing_principles = not args.no_reuse_existing_principles
    for split_name, split_ds in dataset_dict.items():
        print(f"Processing split: {split_name} rows={split_ds.num_rows}")
        rows = prepare_rows(split_ds, reuse_existing_principles)
        if not rows:
            print(f"No rows kept for split {split_name}")
            built_splits[split_name] = Dataset.from_list([])
            saved_rows_by_split[split_name] = []
            split_reports[split_name] = summarize_rows([])
            continue

        build_initial_principles(rows, principle_template, llm, tokenizer, args)
        run_auditor(rows, auditor_template, llm, tokenizer, args)
        run_revisor(rows, revisor_template, llm, tokenizer, args)
        run_chooser(rows, chooser_template, llm, tokenizer, args)

        saved_rows_by_split[split_name] = rows
        built_splits[split_name] = Dataset.from_list(rows)
        split_reports[split_name] = summarize_rows(rows)
        print(f"Built refined labels for split {split_name}: {len(rows)} rows")

    if args.output_dataset_path:
        output_dataset_path = resolve_repo_file(args.output_dataset_path)
        output_dataset_path.parent.mkdir(parents=True, exist_ok=True)
        if len(built_splits) == 1 and not args.split:
            next(iter(built_splits.values())).save_to_disk(str(output_dataset_path))
        else:
            DatasetDict(built_splits).save_to_disk(str(output_dataset_path))
        print(f"Saved refined dataset to {output_dataset_path}")

    if args.output_dir:
        output_dir = resolve_repo_file(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        split_name = args.split or next(iter(saved_rows_by_split))
        rows = saved_rows_by_split[split_name]
        report = dict(split_reports[split_name])
        report.update(
            {
                "title": "Self-refined principle labels from all correct traces",
                "input_dataset_path": args.input_dataset_path,
                "split": split_name,
                "base_model": args.base_model,
                "principle_prompt_path": args.principle_prompt_path,
                "auditor_prompt_path": args.auditor_prompt_path,
                "revisor_prompt_path": args.revisor_prompt_path,
                "chooser_prompt_path": args.chooser_prompt_path,
                "max_model_len": args.max_model_len,
                "max_num_batched_tokens": args.max_num_batched_tokens,
                "gpu_memory_utilization": args.gpu_memory_utilization,
                "reuse_existing_principles": reuse_existing_principles,
            }
        )
        write_json(output_dir / "report.json", report)
        if args.write_rows_json:
            write_json(output_dir / "rows.json", rows)
        write_json(output_dir / "sample_rows.json", rows[: min(20, len(rows))])
        write_summary_md(output_dir / "summary.md", report, rows)
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
