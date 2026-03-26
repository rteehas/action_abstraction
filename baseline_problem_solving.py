from __future__ import annotations

import json
from argparse import ArgumentParser
from pathlib import Path
from statistics import mean

from datasets import Dataset, DatasetDict, load_from_disk
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from process_deepscaler_dataset import extract_answer_from_solution, parse, verify


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = "/deepscaler_qwne1_7B_solutions_scored"
DEFAULT_MODEL = "Qwen/Qwen3-1.7B"
DEFAULT_PROMPT_TEMPLATE = REPO_ROOT / "prompt_templates" / "baseline_problem_solving.txt"


def make_problem_solving_prompt(problem: str, prompt_template: str, cheatsheet: str = "") -> str:
    rendered = prompt_template
    rendered = rendered.replace("{{PROBLEM}}", problem)
    rendered = rendered.replace("{problem_description}", problem)
    rendered = rendered.replace("{cheatsheet}", cheatsheet)
    return rendered


def parse_args() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--input_dataset_path", type=str, default=DEFAULT_DATASET)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--base_model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--prompt_template_path", type=str, default=str(DEFAULT_PROMPT_TEMPLATE))
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
    parser.add_argument("--cheatsheet_text", type=str, default="")
    return parser


def resolve_template(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def load_dataset_rows(path: str) -> Dataset:
    dataset = load_from_disk(path)
    if isinstance(dataset, DatasetDict):
        if "train" in dataset:
            dataset = dataset["train"]
        else:
            dataset = next(iter(dataset.values()))
    assert isinstance(dataset, Dataset)
    return dataset


def select_shard(dataset: Dataset, shard: int, num_shards: int, max_rows: int | None) -> Dataset:
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


def score_answer(expected_answer: str, predicted_boxed_answer: str | None) -> bool:
    if predicted_boxed_answer is None:
        return False
    target = f"\\boxed{{{expected_answer}}}"
    try:
        return bool(verify(parse(target), parse(predicted_boxed_answer)))
    except Exception:
        return False


def main() -> None:
    args = parse_args().parse_args()
    print("Args:")
    print(args)

    prompt_template = resolve_template(args.prompt_template_path).read_text(encoding="utf-8")
    dataset = load_dataset_rows(args.input_dataset_path)
    shard_rows = select_shard(dataset, args.shard, args.num_shards, args.max_rows)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    prompts: list[str] = []
    row_ids: list[int] = []

    for local_idx, example in enumerate(shard_rows):
        problem = example["problem"]
        prompt = make_problem_solving_prompt(problem, prompt_template, args.cheatsheet_text)
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=not args.disable_thinking,
        )
        prompts.append(text)
        row_ids.append(local_idx)

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

    outputs = llm.generate(prompts, sampling_params)

    additional_solutions: list[list[str]] = []
    additional_answers: list[list[str | None]] = []
    additional_num_correct: list[int] = []
    additional_passrate: list[float] = []
    augmented_solutions: list[list[str]] = []
    augmented_answers: list[list[str | None]] = []
    augmented_num_correct: list[int] = []
    augmented_passrate: list[float] = []

    for example, output in zip(shard_rows, outputs):
        new_solutions = [candidate.text for candidate in output.outputs]
        new_answers = [extract_answer_from_solution(text) for text in new_solutions]
        new_correct = sum(score_answer(example["answer"], answer) for answer in new_answers)

        existing_solutions = list(example.get("generated_solution") or [])
        existing_answers = list(example.get("generated_answer") or [])
        if existing_solutions and not existing_answers:
            existing_answers = [extract_answer_from_solution(text) for text in existing_solutions]

        merged_solutions = existing_solutions + new_solutions
        merged_answers = existing_answers + new_answers
        merged_correct = sum(score_answer(example["answer"], answer) for answer in merged_answers)

        additional_solutions.append(new_solutions)
        additional_answers.append(new_answers)
        additional_num_correct.append(new_correct)
        additional_passrate.append(new_correct / len(new_answers) if new_answers else 0.0)
        augmented_solutions.append(merged_solutions)
        augmented_answers.append(merged_answers)
        augmented_num_correct.append(merged_correct)
        augmented_passrate.append(merged_correct / len(merged_answers) if merged_answers else 0.0)

    augmented_dataset = shard_rows.add_column("additional_generated_solution", additional_solutions)
    augmented_dataset = augmented_dataset.add_column("additional_generated_answer", additional_answers)
    augmented_dataset = augmented_dataset.add_column("additional_num_correct", additional_num_correct)
    augmented_dataset = augmented_dataset.add_column("additional_passrate", additional_passrate)
    augmented_dataset = augmented_dataset.add_column("augmented_generated_solution", augmented_solutions)
    augmented_dataset = augmented_dataset.add_column("augmented_generated_answer", augmented_answers)
    augmented_dataset = augmented_dataset.add_column("augmented_num_correct", augmented_num_correct)
    augmented_dataset = augmented_dataset.add_column("augmented_passrate", augmented_passrate)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dataset_path = output_dir / "dataset"
    augmented_dataset.save_to_disk(str(output_dataset_path))

    report = {
        "title": "Baseline problem solving resample",
        "input_dataset_path": args.input_dataset_path,
        "output_dataset_path": str(output_dataset_path),
        "base_model": args.base_model,
        "prompt_template_path": str(resolve_template(args.prompt_template_path)),
        "seed": args.seed,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "n_solutions": args.n_solutions,
        "max_tokens": args.max_tokens,
        "enable_thinking": not args.disable_thinking,
        "cheatsheet_text": args.cheatsheet_text,
        "num_rows": augmented_dataset.num_rows,
        "shard": args.shard,
        "num_shards": args.num_shards,
        "mean_original_passrate": mean(float(example["passrate"]) for example in shard_rows) if shard_rows.num_rows else 0.0,
        "mean_additional_passrate": mean(additional_passrate) if additional_passrate else 0.0,
        "mean_augmented_passrate": mean(augmented_passrate) if augmented_passrate else 0.0,
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    sample_rows = []
    for idx in range(min(5, augmented_dataset.num_rows)):
        row = augmented_dataset[idx]
        sample_rows.append(
            {
                "row_idx": idx,
                "problem": row["problem"],
                "answer": row["answer"],
                "original_passrate": row["passrate"],
                "additional_passrate": row["additional_passrate"],
                "augmented_passrate": row["augmented_passrate"],
                "additional_generated_answer": row["additional_generated_answer"],
            }
        )
    (output_dir / "sample_rows.json").write_text(json.dumps(sample_rows, indent=2), encoding="utf-8")

    print(f"saved dataset to {output_dataset_path}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
