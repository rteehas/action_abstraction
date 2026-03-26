#!/usr/bin/env python
"""Build a VERL-compatible two-policy eval parquet from an on-disk dataset.

The output parquet uses the standard RL schema and preserves the original
``source`` field as ``data_source`` so validation metrics can be reported
separately for AIME and AMC.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_from_disk


DEFAULT_SOURCE = "/workspace/aime2025_amc2023_eval_set"
DEFAULT_TEMPLATE = "/workspace/action_abstraction/prompt_templates/sft_principle_generation.txt"
DEFAULT_OUTPUT = "/workspace/action_abstraction/verl_data/two_policy_aime2025_amc2023_eval"


def load_template(path: str) -> str:
    return Path(path).read_text()


def render_template(template: str, problem: str) -> str:
    return template.replace("{{PROBLEM}}", problem)


def build_rows(dataset, template: str, source_dataset: str) -> tuple[list[dict], Counter]:
    rows = []
    source_counts = Counter()
    for source_index, source_row in enumerate(dataset):
        source = str(source_row["source"]).strip().lower()
        source_counts[source] += 1
        rows.append(
            {
                "problem": source_row["problem"],
                "answer": source_row["answer"],
                "prompt": [{"role": "user", "content": render_template(template, source_row["problem"])}],
                "data_source": source,
                "ability": "math",
                "reward_model": {
                    "ground_truth": source_row["answer"],
                    "style": "rule",
                },
                "extra_info": {
                    "source_index": int(source_index),
                    "source_dataset": source_dataset,
                    "source": source,
                },
            }
        )
    return rows, source_counts


def write_parquet(rows: list[dict], path: Path) -> None:
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dataset", default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
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

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_from_disk(args.source_dataset)
    template = load_template(args.prompt_template)
    rows, source_counts = build_rows(dataset=dataset, template=template, source_dataset=args.source_dataset)

    val_path = output_dir / "val.parquet"
    write_parquet(rows, val_path)

    summary = {
        "source_dataset": args.source_dataset,
        "output_dir": str(output_dir),
        "prompt_template": args.prompt_template,
        "total_rows": len(rows),
        "source_counts": dict(source_counts),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(f"Wrote {len(rows)} rows to {val_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
