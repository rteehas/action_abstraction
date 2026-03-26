from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

from datasets import Dataset, DatasetDict, load_from_disk


def load_rows(path: str) -> list[dict]:
    dataset = load_from_disk(path)
    if isinstance(dataset, DatasetDict):
        if 'train' in dataset:
            dataset = dataset['train']
        else:
            dataset = next(iter(dataset.values()))
    assert isinstance(dataset, Dataset)
    return [dict(dataset[i]) for i in range(len(dataset))]


def summarize_distribution(values: list[float]) -> dict[str, int]:
    counts = {}
    for value in sorted(set(values)):
        counts[str(value)] = sum(1 for x in values if x == value)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_manifest', required=True)
    parser.add_argument('--baseline_dataset', required=True)
    parser.add_argument('--conditioned_datasets', nargs='+', required=True)
    parser.add_argument('--output_dir', required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(Path(args.sample_manifest).read_text(encoding='utf-8'))
    baseline_rows = load_rows(args.baseline_dataset)
    baseline_by_key = {row['sample_idx']: row for row in baseline_rows}

    summary = {
        'sample_size': len(manifest),
        'baseline_mean_passrate': mean(float(row['additional_passrate']) for row in baseline_rows),
        'baseline_distribution': summarize_distribution([float(row['additional_passrate']) for row in baseline_rows]),
        'temperatures': [],
    }

    row_report = {entry['sample_idx']: {
        'sample_idx': entry['sample_idx'],
        'dataset_idx': entry['dataset_idx'],
        'source': entry['source'],
        'dataset_passrate': entry['passrate'],
        'problem': entry['problem'],
        'answer': entry['answer'],
        'baseline_passrate': float(baseline_by_key[entry['sample_idx']]['additional_passrate']),
        'baseline_generated_answer': baseline_by_key[entry['sample_idx']]['additional_generated_answer'],
    } for entry in manifest}

    for conditioned_path in args.conditioned_datasets:
        rows = load_rows(conditioned_path)
        by_sample = defaultdict(list)
        for row in rows:
            by_sample[row['sample_idx']].append(row)
        temperature = None
        conditioned_passrates = []
        best_abstraction_passrates = []
        improved = same = regressed = 0
        for sample_idx, group in sorted(by_sample.items()):
            temperature = group[0]['abstraction_temperature']
            abstraction_passrates = [float(row['conditioned_passrate']) for row in group]
            aggregate = mean(abstraction_passrates)
            best = max(abstraction_passrates)
            baseline = row_report[sample_idx]['baseline_passrate']
            conditioned_passrates.append(aggregate)
            best_abstraction_passrates.append(best)
            if aggregate > baseline:
                improved += 1
            elif aggregate < baseline:
                regressed += 1
            else:
                same += 1
            row_report[sample_idx].setdefault('temperature_results', {})[str(temperature)] = {
                'aggregate_conditioned_passrate': aggregate,
                'best_abstraction_passrate': best,
                'abstractions': [
                    {
                        'abstraction_idx': row['abstraction_idx'],
                        'generated_principles_text': row['generated_principles_text'],
                        'conditioned_passrate': float(row['conditioned_passrate']),
                        'conditioned_generated_answer': row['conditioned_generated_answer'],
                    }
                    for row in sorted(group, key=lambda r: r['abstraction_idx'])
                ],
            }
        summary['temperatures'].append({
            'temperature': temperature,
            'mean_conditioned_passrate': mean(conditioned_passrates),
            'mean_best_abstraction_passrate': mean(best_abstraction_passrates),
            'distribution': summarize_distribution(conditioned_passrates),
            'improved_count': improved,
            'same_count': same,
            'regressed_count': regressed,
        })

    (output_dir / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    (output_dir / 'rows.json').write_text(json.dumps([row_report[k] for k in sorted(row_report)], indent=2), encoding='utf-8')

    lines = [
        '# RLAD Sample Sweep Summary',
        '',
        f"- sample size: `{summary['sample_size']}`",
        f"- baseline mean passrate: `{summary['baseline_mean_passrate']:.4f}`",
        f"- baseline distribution: `{summary['baseline_distribution']}`",
        '',
        '## Temperatures',
        '',
    ]
    for result in summary['temperatures']:
        lines.extend([
            f"### Temperature {result['temperature']}",
            f"- mean conditioned passrate: `{result['mean_conditioned_passrate']:.4f}`",
            f"- mean best-abstraction passrate: `{result['mean_best_abstraction_passrate']:.4f}`",
            f"- improved / same / regressed: `{result['improved_count']}` / `{result['same_count']}` / `{result['regressed_count']}`",
            f"- distribution: `{result['distribution']}`",
            '',
        ])
    (output_dir / 'summary.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
