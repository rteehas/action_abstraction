from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any

from datasets import Dataset, DatasetDict, load_from_disk
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from process_deepscaler_dataset import extract_answer_from_solution, parse, verify


DEFAULT_BASE_MODEL = "Qwen/Qwen3-1.7B"
DEFAULT_PROMPT_TEMPLATE = "prompt_templates/hint_conditioned_problem_solving_rich_v1.txt"
DEFAULT_RESULT_PREFIX = "conditioned"
DEFAULT_PRINCIPLE_FIELDS = ("generated_principles_text", "principles", "abstraction")
DEFAULT_REFERENCE_PASSRATE_FIELDS = ("baseline_passrate", "passrate")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--base_model", type=str, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--prompt_template_path", type=str, default=DEFAULT_PROMPT_TEMPLATE)
    parser.add_argument("--problem_field", type=str, default="problem")
    parser.add_argument("--answer_field", type=str, default="answer")
    parser.add_argument("--principle_field", type=str, default="")
    parser.add_argument("--reference_passrate_field", type=str, default="")
    parser.add_argument("--result_prefix", type=str, default=DEFAULT_RESULT_PREFIX)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--n_solutions", type=int, default=4)
    parser.add_argument("--max_tokens", type=int, default=8000)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--max_num_batched_tokens", type=int, default=4096)
    parser.add_argument("--disable_thinking", action="store_true")
    parser.add_argument("--store_prompt_text", action="store_true")
    return parser


def resolve_repo_file(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def load_rows(path_str: str) -> tuple[list[dict[str, Any]], str]:
    path = Path(path_str)
    if path.is_dir():
        dataset = load_from_disk(str(path))
        if isinstance(dataset, DatasetDict):
            if "train" in dataset:
                dataset = dataset["train"]
            else:
                dataset = next(iter(dataset.values()))
        assert isinstance(dataset, Dataset)
        return [dict(example) for example in dataset], "dataset"

    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return [dict(example) for example in payload], "json"
        if isinstance(payload, dict):
            if "rows" in payload and isinstance(payload["rows"], list):
                return [dict(example) for example in payload["rows"]], "json"
            if "train_rows" in payload or "val_rows" in payload:
                rows: list[dict[str, Any]] = []
                for split_name in ("train_rows", "val_rows"):
                    split_rows = payload.get(split_name) or []
                    split = "train" if split_name == "train_rows" else "val"
                    for example in split_rows:
                        copied = dict(example)
                        copied.setdefault("split", split)
                        rows.append(copied)
                return rows, "manifest"
        raise ValueError(f"Unsupported JSON payload at {path}")

    raise ValueError(f"Unsupported input path: {path}")


def select_shard(rows: list[dict[str, Any]], shard: int, num_shards: int, max_rows: int | None) -> list[dict[str, Any]]:
    if num_shards < 1:
        raise ValueError("num_shards must be >= 1")
    if shard < 0 or shard >= num_shards:
        raise ValueError("shard must satisfy 0 <= shard < num_shards")
    shard_size = len(rows) // num_shards
    start = shard * shard_size
    end = len(rows) if shard == num_shards - 1 else (shard + 1) * shard_size
    selected = rows[start:end]
    if max_rows is not None:
        selected = selected[:max_rows]
    return selected


def choose_principle_field(rows: list[dict[str, Any]], requested_field: str) -> str:
    if requested_field:
        return requested_field
    for field in DEFAULT_PRINCIPLE_FIELDS:
        if any((example.get(field) or "").strip() for example in rows):
            return field
    raise ValueError(
        "No non-empty principle field found. Tried: " + ", ".join(DEFAULT_PRINCIPLE_FIELDS)
    )


def choose_reference_passrate_field(rows: list[dict[str, Any]], requested_field: str) -> str | None:
    if requested_field:
        return requested_field
    for field in DEFAULT_REFERENCE_PASSRATE_FIELDS:
        if any(field in example for example in rows):
            return field
    return None


def score_answer(expected_answer: str, predicted_boxed_answer: str | None) -> bool:
    if predicted_boxed_answer is None:
        return False
    target = f"\\boxed{{{expected_answer}}}"
    try:
        return bool(verify(parse(target), parse(predicted_boxed_answer)))
    except Exception:
        return False


def render_conditioned_prompt(problem: str, principle: str, template: str) -> str:
    rendered = template
    rendered = rendered.replace("{{PROBLEM}}", problem)
    rendered = rendered.replace("{{ABSTRACTION}}", principle)
    rendered = rendered.replace("{problem_description}", problem)
    rendered = rendered.replace("{cheatsheet}", principle)
    return rendered


def summarize_distribution(values: list[float]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in sorted(set(values)):
        counts[str(value)] = sum(1 for candidate in values if candidate == value)
    return counts


def write_summary_md(path: Path, report: dict[str, Any], rows: list[dict[str, Any]], prefix: str) -> None:
    lines = [
        "# Principle-Conditioned Problem Solving",
        "",
        f"- input path: `{report['input_path']}`",
        f"- input format: `{report['input_format']}`",
        f"- output path: `{report['output_path']}`",
        f"- base model: `{report['base_model']}`",
        f"- prompt template: `{report['prompt_template_path']}`",
        f"- principle field: `{report['principle_field']}`",
        f"- reference passrate field: `{report['reference_passrate_field']}`",
        f"- enable thinking: `{report['enable_thinking']}`",
        f"- num rows: `{report['num_rows']}`",
        f"- prompt rows: `{report['num_prompt_rows']}`",
        f"- empty principle rows: `{report['empty_principle_rows']}`",
        f"- mean {prefix} passrate: `{report[f'mean_{prefix}_passrate']:.6f}`",
    ]

    if report.get("mean_reference_passrate") is not None:
        lines.append(f"- mean reference passrate: `{report['mean_reference_passrate']:.6f}`")
        lines.append(f"- improved / same / regressed: `{report['improved_count']}` / `{report['same_count']}` / `{report['regressed_count']}`")

    lines.extend([
        "",
        "## Passrate Distribution",
        "",
    ])
    for key, value in report[f"{prefix}_passrate_distribution"].items():
        lines.append(f"- {key}: {value}")

    lines.extend([
        "",
        "## Sample Rows",
        "",
    ])

    for row in rows[: min(5, len(rows))]:
        lines.extend(
            [
                f"### Row {row.get('row_id', '(no row_id)')}",
                row.get("problem", ""),
                "",
                f"#### Principle ({report['principle_field']})",
                row.get(report["principle_field"], "") or "(empty)",
                "",
                f"#### {prefix.title()} Passrate",
                str(row.get(f"{prefix}_passrate")),
                "",
            ]
        )

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    print("Args:")
    print(args)

    rows, input_format = load_rows(args.input_path)
    rows = select_shard(rows, args.shard, args.num_shards, args.max_rows)
    if not rows:
        raise ValueError("No rows selected after applying shard/max_rows")

    principle_field = choose_principle_field(rows, args.principle_field)
    reference_passrate_field = choose_reference_passrate_field(rows, args.reference_passrate_field)
    prompt_template = resolve_repo_file(args.prompt_template_path).read_text(encoding="utf-8")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    prompts: list[str] = []
    prompt_row_indices: list[int] = []

    prefix = args.result_prefix
    empty_principle_rows = 0

    for idx, row in enumerate(rows):
        row.setdefault("row_id", idx)
        principle = (row.get(principle_field) or "").strip()
        row[f"{prefix}_principle_field"] = principle_field
        if not principle:
            empty_principle_rows += 1
            row[f"{prefix}_generated_solution"] = []
            row[f"{prefix}_generated_answer"] = []
            row[f"{prefix}_num_correct"] = 0
            row[f"{prefix}_passrate"] = 0.0
            if args.store_prompt_text:
                row[f"{prefix}_prompt_text"] = ""
            continue

        prompt = render_conditioned_prompt(
            str(row[args.problem_field]),
            principle,
            prompt_template,
        )
        if args.store_prompt_text:
            row[f"{prefix}_prompt_text"] = prompt
        prompts.append(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=not args.disable_thinking,
            )
        )
        prompt_row_indices.append(idx)

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        n=args.n_solutions,
        seed=args.seed,
    )

    llm = LLM(
        model=args.base_model,
        dtype="bfloat16",
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_batched_tokens=args.max_num_batched_tokens,
    )

    outputs = llm.generate(prompts, sampling_params) if prompts else []

    for row_idx, output in zip(prompt_row_indices, outputs):
        row = rows[row_idx]
        generated_solutions = [candidate.text for candidate in output.outputs]
        generated_answers = [extract_answer_from_solution(text) for text in generated_solutions]
        num_correct = sum(score_answer(str(row[args.answer_field]), answer) for answer in generated_answers)
        row[f"{prefix}_generated_solution"] = generated_solutions
        row[f"{prefix}_generated_answer"] = generated_answers
        row[f"{prefix}_num_correct"] = num_correct
        row[f"{prefix}_passrate"] = num_correct / len(generated_answers) if generated_answers else 0.0

    conditioned_passrates = [float(row.get(f"{prefix}_passrate", 0.0)) for row in rows]

    report: dict[str, Any] = {
        "title": "Principle-conditioned problem solving",
        "input_path": args.input_path,
        "input_format": input_format,
        "base_model": args.base_model,
        "prompt_template_path": str(resolve_repo_file(args.prompt_template_path)),
        "principle_field": principle_field,
        "reference_passrate_field": reference_passrate_field,
        "result_prefix": prefix,
        "seed": args.seed,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "n_solutions": args.n_solutions,
        "max_tokens": args.max_tokens,
        "enable_thinking": not args.disable_thinking,
        "num_rows": len(rows),
        "num_prompt_rows": len(prompts),
        "empty_principle_rows": empty_principle_rows,
        f"mean_{prefix}_passrate": mean(conditioned_passrates) if conditioned_passrates else 0.0,
        f"{prefix}_passrate_distribution": summarize_distribution(conditioned_passrates),
        "shard": args.shard,
        "num_shards": args.num_shards,
    }

    if reference_passrate_field is not None:
        reference_passrates = [float(row.get(reference_passrate_field, 0.0)) for row in rows]
        improved_count = sum(1 for row in rows if float(row.get(f"{prefix}_passrate", 0.0)) > float(row.get(reference_passrate_field, 0.0)))
        same_count = sum(1 for row in rows if float(row.get(f"{prefix}_passrate", 0.0)) == float(row.get(reference_passrate_field, 0.0)))
        regressed_count = len(rows) - improved_count - same_count
        report["mean_reference_passrate"] = mean(reference_passrates) if reference_passrates else None
        report["improved_count"] = improved_count
        report["same_count"] = same_count
        report["regressed_count"] = regressed_count

    output_dir = resolve_repo_file(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if input_format == "dataset":
        output_path = output_dir / "dataset"
        Dataset.from_list(rows).save_to_disk(str(output_path))
    else:
        output_path = output_dir / "rows.json"
        output_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    report["output_path"] = str(output_path)
    (output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    sample_rows = []
    for row in rows[: min(5, len(rows))]:
        sample_rows.append(
            {
                "row_id": row.get("row_id"),
                "problem": row.get(args.problem_field),
                "answer": row.get(args.answer_field),
                "principle_field": principle_field,
                "principle_excerpt": (row.get(principle_field) or "")[:800],
                "reference_passrate": row.get(reference_passrate_field) if reference_passrate_field else None,
                f"{prefix}_passrate": row.get(f"{prefix}_passrate"),
                f"{prefix}_generated_answer": row.get(f"{prefix}_generated_answer"),
            }
        )
    (output_dir / "sample_rows.json").write_text(json.dumps(sample_rows, indent=2), encoding="utf-8")
    write_summary_md(output_dir / "summary.md", report, rows, prefix)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
