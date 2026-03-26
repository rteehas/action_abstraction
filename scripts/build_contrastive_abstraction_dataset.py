from __future__ import annotations

import argparse
import random
from pathlib import Path

from datasets import Dataset, DatasetDict, load_from_disk

from contrastive_abstraction_utils import (
    classify_generated_answers,
    format_solution_block,
    pick_shortest,
    read_text,
    repo_path,
)


DEFAULT_INPUT = "/deepscaler_qwne1_7B_solutions_scored"
DEFAULT_OUTPUT = "contrastive_abstraction_datasets/deepscaler_mixed_correct_incorrect"
DEFAULT_PROMPT = "prompt_templates/contrastive_abstraction_labeling_prompt.txt"
DEFAULT_FEW_SHOTS = "prompt_templates/contrastive_abstraction_labeling_few_shots.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default=DEFAULT_INPUT)
    parser.add_argument("--output_path", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--prompt_path", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--few_shots_path", type=str, default=DEFAULT_FEW_SHOTS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--max_incorrect", type=int, default=2)
    parser.add_argument("--mixed_only", action="store_true", default=True)
    parser.add_argument("--allow_all_correct", action="store_true")
    parser.add_argument("--correct_selection", choices=["shortest", "first", "all"], default="shortest")
    parser.add_argument("--max_solution_chars", type=int, default=6000)
    return parser.parse_args()


def resolve_repo_file(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return repo_path(path_str)


def build_prompt(template: str, few_shots: str, example: dict, max_solution_chars: int) -> str:
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
    return (
        template.replace("{{FEW_SHOT_EXAMPLES}}", few_shots)
        .replace("{{PROBLEM}}", example["problem"])
        .replace("{{CORRECT_SOLUTIONS_BLOCK}}", correct_block)
        .replace("{{INCORRECT_SOLUTIONS_BLOCK}}", incorrect_block)
    )


def select_correct_pairs(correct_pairs: list[tuple[int, str]], mode: str) -> list[tuple[int, str]]:
    if mode == "all":
        return correct_pairs
    if mode == "first":
        return [correct_pairs[0]]
    shortest = pick_shortest([solution for _, solution in correct_pairs])
    for pair in correct_pairs:
        if pair[1] == shortest:
            return [pair]
    return [correct_pairs[0]]


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    dataset = load_from_disk(args.input_path)
    if isinstance(dataset, DatasetDict):
        dataset = dataset["train"]

    template = read_text(resolve_repo_file(args.prompt_path))
    few_shots = read_text(resolve_repo_file(args.few_shots_path)) if args.few_shots_path else ""

    rows = []
    for idx, ex in enumerate(dataset):
        generated_solutions = list(ex["generated_solution"])
        generated_answers = list(ex["generated_answer"])
        correctness = classify_generated_answers(ex["answer"], generated_answers)

        correct_pairs = [(i, generated_solutions[i]) for i, is_correct in enumerate(correctness) if is_correct]
        incorrect_pairs = [(i, generated_solutions[i]) for i, is_correct in enumerate(correctness) if not is_correct]

        if not correct_pairs:
            continue
        if args.mixed_only and not incorrect_pairs and not args.allow_all_correct:
            continue

        selected_incorrect_pairs = incorrect_pairs[: args.max_incorrect]
        if not selected_incorrect_pairs and not args.allow_all_correct:
            continue

        selected_correct_pairs = select_correct_pairs(correct_pairs, args.correct_selection)

        row = {
            "row_id": idx,
            "problem": ex["problem"],
            "answer": ex["answer"],
            "reference_solution": ex["solution"],
            "generated_solution": generated_solutions,
            "generated_answer": generated_answers,
            "solution_correctness": correctness,
            "num_correct": int(sum(correctness)),
            "passrate": float(sum(correctness) / len(correctness)) if correctness else 0.0,
            "selected_correct_indices": [pair_idx for pair_idx, _ in selected_correct_pairs],
            "selected_incorrect_indices": [i for i, _ in selected_incorrect_pairs],
            "selected_correct_solutions": [solution for _, solution in selected_correct_pairs],
            "selected_incorrect_solutions": [solution for _, solution in selected_incorrect_pairs],
        }
        row["contrastive_prompt"] = build_prompt(
            template=template,
            few_shots=few_shots,
            example=row,
            max_solution_chars=args.max_solution_chars,
        )
        rows.append(row)
        if args.max_rows is not None and len(rows) >= args.max_rows:
            break

    built = Dataset.from_list(rows)
    if built.num_rows > 1:
        split = built.train_test_split(test_size=args.val_frac, seed=args.seed)
    else:
        split = DatasetDict({"train": built, "test": built})
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    split.save_to_disk(str(output_path))

    print(f"Saved contrastive dataset to {output_path}")
    print(split)
    for split_name, split_ds in split.items():
        if split_ds.num_rows == 0:
            continue
        mean_num_correct = sum(split_ds["num_correct"]) / split_ds.num_rows
        print(f"{split_name}: rows={split_ds.num_rows} mean_num_correct={mean_num_correct:.3f}")
        print("Sample prompt preview:")
        print(split_ds[0]["contrastive_prompt"][:1500])


if __name__ == "__main__":
    main()
