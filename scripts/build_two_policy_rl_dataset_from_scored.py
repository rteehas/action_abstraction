#!/usr/bin/env python
"""Build a VERL-compatible two-policy RL train dataset from a scored dataset on disk.

The output parquet includes a standard ``prompt`` column for dataset compatibility.
TwoPolicyGRPOTrainer still rebuilds abstraction rollout prompts from
``two_policy.abstraction_prompt_template_path`` at training time, so the prompt
rendered here is mainly for schema compatibility and dataset-side preprocessing.
"""

import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_from_disk


DEFAULT_SOURCE = "/deepscaler_qwne1_7B_solutions_scored"
DEFAULT_TEMPLATE = "/workspace/action_abstraction/prompt_templates/sft_principle_generation.txt"
DEFAULT_OUTPUT = "/workspace/action_abstraction/verl_data/two_policy_deepscaler_qwne1_7b_passrate_025_075"


def load_template(path: str) -> str:
    return Path(path).read_text()


def render_template(template: str, problem: str) -> str:
    return template.replace("{{PROBLEM}}", problem)


def build_rows(dataset, template: str, min_passrate: float, max_passrate: float, source_dataset: str) -> tuple[list[dict], int]:
    rows = []
    total_rows = len(dataset)
    for source_index, source_row in enumerate(dataset):
        passrate = float(source_row["passrate"])
        if not (min_passrate <= passrate <= max_passrate):
            continue

        answer = source_row["answer"]
        prompt_text = render_template(template, source_row["problem"])
        rows.append(
            {
                "problem": source_row["problem"],
                "answer": answer,
                "prompt": [{"role": "user", "content": prompt_text}],
                "data_source": "deepscaler_qwne1_7b_scored",
                "ability": "math",
                "reward_model": {
                    "ground_truth": answer,
                    "style": "rule",
                },
                "extra_info": {
                    "source_index": int(source_index),
                    "source_dataset": source_dataset,
                    "passrate": passrate,
                    "num_correct": int(source_row["num_correct"]),
                },
            }
        )
    return rows, total_rows


def write_parquet(rows: list[dict], path: Path) -> None:
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dataset", default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--min-passrate", type=float, default=0.25)
    parser.add_argument("--max-passrate", type=float, default=0.75)
    parser.add_argument(
        "--prompt-template",
        default=DEFAULT_TEMPLATE,
        help=(
            "Template used to populate the parquet compatibility prompt column. "
            "TwoPolicyGRPOTrainer rebuilds abstraction rollout prompts from "
            "two_policy.abstraction_prompt_template_path, so override the trainer "
            "config as well if you want rollout behavior to change."
        ),
    )
    args = parser.parse_args()

    if args.min_passrate > args.max_passrate:
        raise ValueError("min-passrate must be <= max-passrate")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_from_disk(args.source_dataset)
    template = load_template(args.prompt_template)
    rows, total_rows = build_rows(
        dataset=dataset,
        template=template,
        min_passrate=args.min_passrate,
        max_passrate=args.max_passrate,
        source_dataset=args.source_dataset,
    )

    train_path = output_dir / "train.parquet"
    write_parquet(rows, train_path)

    summary = {
        "source_dataset": args.source_dataset,
        "output_dir": str(output_dir),
        "prompt_template": args.prompt_template,
        "min_passrate": args.min_passrate,
        "max_passrate": args.max_passrate,
        "total_source_rows": total_rows,
        "kept_rows": len(rows),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(f"Wrote {len(rows)} rows to {train_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
