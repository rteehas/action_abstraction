from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
import random
import sys

from datasets import Dataset, DatasetDict, load_from_disk

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def repo_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', required=True)
    parser.add_argument('--output_path', required=True)
    parser.add_argument('--selection', choices=['improved_only', 'non_regressed'], default='non_regressed')
    parser.add_argument('--split_seed', type=int, default=100)
    parser.add_argument('--test_frac', type=float, default=0.1)
    return parser


def keep_example(example: dict, selection: str) -> bool:
    baseline = float(example['baseline_passrate'])
    conditioned = float(example['conditioned_passrate'])
    if selection == 'improved_only':
        return conditioned > baseline
    return conditioned >= baseline


def stratified_split(rows: list[dict], seed: int, test_frac: float) -> tuple[list[dict], list[dict]]:
    buckets: dict[float, list[dict]] = defaultdict(list)
    for row in rows:
        buckets[float(row['baseline_passrate'])].append(row)

    rng = random.Random(seed)
    train_rows: list[dict] = []
    test_rows: list[dict] = []
    for _, bucket_rows in sorted(buckets.items()):
        bucket = list(bucket_rows)
        rng.shuffle(bucket)
        test_size = max(1, int(round(len(bucket) * test_frac))) if len(bucket) > 1 else 1
        test_size = min(test_size, len(bucket) - 1) if len(bucket) > 1 else 1
        test_rows.extend(bucket[:test_size])
        train_rows.extend(bucket[test_size:])
    return train_rows, test_rows


def main() -> None:
    args = build_parser().parse_args()

    input_path = repo_path(args.input_path)
    output_path = repo_path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = load_from_disk(str(input_path))
    rows = [dict(dataset[i]) for i in range(len(dataset))]
    selected_rows = [row for row in rows if keep_example(row, args.selection)]

    for row in selected_rows:
        row['abstraction'] = row['generated_principles_text']

    train_rows, test_rows = stratified_split(selected_rows, seed=args.split_seed, test_frac=args.test_frac)
    dataset_dict = DatasetDict({
        'train': Dataset.from_list(train_rows),
        'test': Dataset.from_list(test_rows),
    })
    dataset_dict.save_to_disk(str(output_path))

    def bucket_counts(split_rows: list[dict]) -> dict[str, int]:
        counts: dict[str, int] = defaultdict(int)
        for row in split_rows:
            counts[str(float(row['baseline_passrate']))] += 1
        return dict(sorted(counts.items(), key=lambda kv: float(kv[0])))

    summary = {
        'input_path': str(input_path),
        'output_path': str(output_path),
        'selection': args.selection,
        'split_seed': args.split_seed,
        'test_frac': args.test_frac,
        'total_rows': len(rows),
        'selected_rows': len(selected_rows),
        'train_rows': len(train_rows),
        'test_rows': len(test_rows),
        'baseline_bucket_counts_total': bucket_counts(selected_rows),
        'baseline_bucket_counts_train': bucket_counts(train_rows),
        'baseline_bucket_counts_test': bucket_counts(test_rows),
    }
    summary_path = output_path.parent / 'summary.json'
    summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
