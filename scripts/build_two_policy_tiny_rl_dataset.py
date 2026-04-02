#!/usr/bin/env python
"""Build a tiny parquet dataset for two-policy smoke/overfit experiments.

This script writes a standard VERL-compatible ``prompt`` column so the output parquet
remains self-contained and usable with generic dataset tooling. In the two-policy
recipe, however, abstraction rollout prompts are rebuilt inside
``TwoPolicyGRPOTrainer`` from ``two_policy.abstraction_prompt_template_path``.
Changing ``--prompt-template`` here therefore affects the parquet payload and
dataset-side preprocessing, but it does not by itself change two-policy rollout
behavior unless the trainer config is overridden to match.
"""

import argparse
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = str(REPO_ROOT / "verl_data" / "sft_dataset_no_rl_partial_05_less1_traintest_concat" / "train.parquet")
DEFAULT_TEMPLATE = str(REPO_ROOT / "prompt_templates" / "sft_principle_generation.txt")


def parse_indices(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def load_template(path: str) -> str:
    return Path(path).read_text()


def render_template(template: str, problem: str) -> str:
    return template.replace("{{PROBLEM}}", problem)


def build_rows(table, indices: list[int], template: str, split: str, source_parquet: str) -> list[dict]:
    rows = []
    for source_index in indices:
        source_row = table.slice(source_index, 1).to_pylist()[0]
        answer = source_row.get("answer")
        reward_model = source_row.get("reward_model") or {}
        ground_truth = reward_model.get("ground_truth") or answer
        data_source = source_row.get("data_source") or source_row.get("source") or "two_policy_tiny"
        ability = source_row.get("ability") or "math"
        prompt_text = render_template(template, source_row["problem"])
        extra_info = dict(source_row.get("extra_info") or {})
        extra_info.update(
            {
                "source_index": int(source_index),
                "split": split,
                "source_parquet": source_parquet,
            }
        )
        rows.append(
            {
                "problem": source_row["problem"],
                "answer": answer,
                "prompt": [{"role": "user", "content": prompt_text}],
                "data_source": data_source,
                "ability": ability,
                "reward_model": {
                    "ground_truth": ground_truth,
                    "style": reward_model.get("style", "rule"),
                },
                "extra_info": extra_info,
            }
        )
    return rows


def write_parquet(rows: list[dict], path: Path) -> None:
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-parquet", default=DEFAULT_SOURCE)
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "verl_data" / "two_policy_tiny_overfit"),
    )
    parser.add_argument("--train-indices", default="144,1606")
    parser.add_argument("--val-indices", default="144,1606")
    parser.add_argument(
        "--prompt-template",
        default=DEFAULT_TEMPLATE,
        help=(
            "Template used to populate the parquet's compatibility prompt column. "
            "TwoPolicyGRPOTrainer rebuilds abstraction rollout prompts from "
            "two_policy.abstraction_prompt_template_path, so override the trainer "
            "config as well if you want rollout behavior to change."
        ),
    )
    args = parser.parse_args()

    source_parquet = Path(args.source_parquet)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    table = pq.read_table(source_parquet)
    template = load_template(args.prompt_template)

    train_rows = build_rows(
        table,
        parse_indices(args.train_indices),
        template,
        split="train",
        source_parquet=str(source_parquet),
    )
    val_rows = build_rows(
        table,
        parse_indices(args.val_indices),
        template,
        split="val",
        source_parquet=str(source_parquet),
    )

    train_path = output_dir / "train.parquet"
    val_path = output_dir / "val.parquet"
    write_parquet(train_rows, train_path)
    write_parquet(val_rows, val_path)

    print(f"Wrote {len(train_rows)} train rows to {train_path}")
    print(f"Wrote {len(val_rows)} val rows to {val_path}")
    for split_name, rows in (("train", train_rows), ("val", val_rows)):
        for row in rows:
            info = row["extra_info"]
            print(
                f"[{split_name}] idx={info['source_index']} data_source={row['data_source']} "
                f"answer={row['reward_model']['ground_truth']} problem={row['problem']}"
            )


if __name__ == "__main__":
    main()
