from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from datasets import Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from process_deepscaler_dataset import extract_answer_from_solution, compute_num_correct


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    rows = json.loads((run_dir / 'rows.json').read_text())
    ds = Dataset.from_list([
        {
            'answer': row['answer'],
            'solver_output_text': row['solver_output_text'] or '',
            'baseline_passrate': row['baseline_passrate'],
            'baseline_num_correct': row['baseline_num_correct'],
        }
        for row in rows
    ])
    ds = ds.map(lambda ex: {'generated_answer': [extract_answer_from_solution(ex['solver_output_text'])]}).map(compute_num_correct)
    num_rows = ds.num_rows
    num_correct = sum(ds['num_correct'])
    baseline_passrate_mean = sum(ds['baseline_passrate']) / num_rows if num_rows else 0.0
    baseline_num_correct_mean = sum(ds['baseline_num_correct']) / num_rows if num_rows else 0.0
    report = {
        'num_rows': num_rows,
        'num_correct': num_correct,
        'accuracy': num_correct / num_rows if num_rows else 0.0,
        'baseline_passrate_mean': baseline_passrate_mean,
        'baseline_num_correct_mean': baseline_num_correct_mean,
        'absolute_accuracy_lift': (num_correct / num_rows if num_rows else 0.0) - baseline_passrate_mean,
        'relative_accuracy_lift': (((num_correct / num_rows) - baseline_passrate_mean) / baseline_passrate_mean) if num_rows and baseline_passrate_mean else None,
        'source': 'process_deepscaler_dataset.py'
    }
    (run_dir / 'rescored_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
