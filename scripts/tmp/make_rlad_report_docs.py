from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--label', required=True)
    args = parser.parse_args()

    root = Path(args.root)
    final_root = root / 'final_summary'
    rows = json.loads((final_root / 'rows.json').read_text())
    summary = json.loads((final_root / 'summary.json').read_text())

    for temp in summary['temperatures']:
        t = str(temp['temperature'])
        by_source = defaultdict(list)
        for row in rows:
            by_source[row['source']].append(row)
        temp['by_source'] = {}
        for source, group in sorted(by_source.items()):
            conditioned = [g['temperature_results'][t]['aggregate_conditioned_passrate'] for g in group]
            best = [g['temperature_results'][t]['best_abstraction_passrate'] for g in group]
            baseline = [g['baseline_passrate'] for g in group]
            improved = sum(c > b for c, b in zip(conditioned, baseline))
            same = sum(c == b for c, b in zip(conditioned, baseline))
            regressed = len(group) - improved - same
            temp['by_source'][source] = {
                'count': len(group),
                'baseline_mean': mean(baseline),
                'mean_conditioned_passrate': mean(conditioned),
                'mean_best_abstraction_passrate': mean(best),
                'improved_count': improved,
                'same_count': same,
                'regressed_count': regressed,
            }

    best = []
    worst = []
    for row in rows:
        for temp, result in row['temperature_results'].items():
            for ab in result['abstractions']:
                entry = {
                    'sample_idx': row['sample_idx'],
                    'dataset_idx': row['dataset_idx'],
                    'source': row['source'],
                    'problem': row['problem'],
                    'answer': row['answer'],
                    'baseline_passrate': row['baseline_passrate'],
                    'temperature': float(temp),
                    'aggregate_conditioned_passrate': result['aggregate_conditioned_passrate'],
                    'best_abstraction_passrate': result['best_abstraction_passrate'],
                    'abstraction_idx': ab['abstraction_idx'],
                    'abstraction_passrate': ab['conditioned_passrate'],
                    'delta_vs_baseline': ab['conditioned_passrate'] - row['baseline_passrate'],
                    'generated_principles_text': ab['generated_principles_text'],
                    'conditioned_generated_answer': ab['conditioned_generated_answer'],
                }
                (best if entry['delta_vs_baseline'] > 0 else worst).append(entry)

    best = sorted(best, key=lambda x: (x['delta_vs_baseline'], x['abstraction_passrate'], x['best_abstraction_passrate']), reverse=True)
    worst = sorted(worst, key=lambda x: (x['delta_vs_baseline'], x['abstraction_passrate'], -x['baseline_passrate']))

    def dedupe(entries, n):
        out = []
        seen = set()
        for e in entries:
            key = (e['sample_idx'], e['temperature'], e['abstraction_idx'])
            if key in seen:
                continue
            seen.add(key)
            out.append(e)
            if len(out) >= n:
                break
        return out

    best = dedupe(best, 8)
    worst = dedupe(worst, 8)

    report_lines = [
        f'# RLAD 12-Problem Sweep Report: {args.label}',
        '',
        '## Setup',
        '',
        '- dataset: `/workspace/rlad_aime_amc_scored`',
        '- sample: 12 problems, balanced `6` AIME + `6` AMC, fixed seed `100`',
        '- baseline solver: `Qwen/Qwen3-1.7B`, `4` samples, `temperature=0.6`, `top_p=0.95`, `top_k=20`, `max_tokens=32768`, thinking on',
        f'- abstraction generator checkpoint: `{args.label}`',
        '- abstraction sweep: temperatures `0.0`, `0.3`, `0.6`, `0.9`, with `4` abstractions per problem',
        '- conditioned solver: same as baseline, with thinking on for all samples',
        '',
        '## Topline',
        '',
        f"- baseline mean passrate: `{summary['baseline_mean_passrate']:.4f}`",
        '',
        '## Temperature Breakdown',
        '',
    ]
    for temp in summary['temperatures']:
        report_lines.extend([
            f"### Temperature {temp['temperature']}",
            f"- mean conditioned passrate: `{temp['mean_conditioned_passrate']:.4f}`",
            f"- mean best-abstraction passrate: `{temp['mean_best_abstraction_passrate']:.4f}`",
            f"- gap between best-of-4 and average: `{temp['mean_best_abstraction_passrate'] - temp['mean_conditioned_passrate']:.4f}`",
            f"- improved / same / regressed vs baseline: `{temp['improved_count']}` / `{temp['same_count']}` / `{temp['regressed_count']}`",
            f"- per-problem distribution: `{temp['distribution']}`",
            '',
            f"Source breakdown for `{temp['temperature']}`:",
            f"- AIME: mean conditioned `{temp['by_source']['aime']['mean_conditioned_passrate']:.4f}`, mean best `{temp['by_source']['aime']['mean_best_abstraction_passrate']:.4f}`, improved/same/regressed `{temp['by_source']['aime']['improved_count']}` / `{temp['by_source']['aime']['same_count']}` / `{temp['by_source']['aime']['regressed_count']}`",
            f"- AMC: mean conditioned `{temp['by_source']['amc']['mean_conditioned_passrate']:.4f}`, mean best `{temp['by_source']['amc']['mean_best_abstraction_passrate']:.4f}`, improved/same/regressed `{temp['by_source']['amc']['improved_count']}` / `{temp['by_source']['amc']['same_count']}` / `{temp['by_source']['amc']['regressed_count']}`",
            '',
        ])

    report_lines.extend([
        '## Takeaways',
        '',
        '- Compare this checkpoint to the final checkpoint by looking at both mean conditioned passrate and mean best-abstraction passrate.',
        '- The average metric tells you whether random abstraction sampling is net-positive; the best-of-4 metric tells you whether a stronger abstraction exists but needs selection.',
        '- The source breakdown shows whether the checkpoint is more helpful on AIME or AMC problems.',
    ])

    examples_lines = [
        f'# Best And Worst Abstractions: {args.label}',
        '',
        'Selection rule:',
        '- best examples are ranked by `abstraction_passrate - baseline_passrate` descending',
        '- worst examples are ranked by the same quantity ascending',
        '- scores are for one abstraction with 4 conditioned solver samples',
        '',
        '## Best Abstractions',
        '',
    ]
    for idx, e in enumerate(best, 1):
        examples_lines.extend([
            f"### Best {idx}",
            f"- row `{e['dataset_idx']}` / sample `{e['sample_idx']}` / source `{e['source']}` / temp `{e['temperature']}` / abstraction `{e['abstraction_idx']}`",
            f"- baseline passrate: `{e['baseline_passrate']:.3f}`",
            f"- abstraction-conditioned passrate: `{e['abstraction_passrate']:.3f}`",
            f"- delta vs baseline: `{e['delta_vs_baseline']:+.3f}`",
            f"- aggregate problem passrate at this temperature: `{e['aggregate_conditioned_passrate']:.3f}`",
            f"- best abstraction passrate at this temperature: `{e['best_abstraction_passrate']:.3f}`",
            f"- conditioned answers: `{e['conditioned_generated_answer']}`",
            '',
            'Problem:',
            e['problem'],
            '',
            'Abstraction:',
            '```text',
            e['generated_principles_text'],
            '```',
            '',
        ])
    examples_lines.extend(['## Worst Abstractions', ''])
    for idx, e in enumerate(worst, 1):
        examples_lines.extend([
            f"### Worst {idx}",
            f"- row `{e['dataset_idx']}` / sample `{e['sample_idx']}` / source `{e['source']}` / temp `{e['temperature']}` / abstraction `{e['abstraction_idx']}`",
            f"- baseline passrate: `{e['baseline_passrate']:.3f}`",
            f"- abstraction-conditioned passrate: `{e['abstraction_passrate']:.3f}`",
            f"- delta vs baseline: `{e['delta_vs_baseline']:+.3f}`",
            f"- aggregate problem passrate at this temperature: `{e['aggregate_conditioned_passrate']:.3f}`",
            f"- best abstraction passrate at this temperature: `{e['best_abstraction_passrate']:.3f}`",
            f"- conditioned answers: `{e['conditioned_generated_answer']}`",
            '',
            'Problem:',
            e['problem'],
            '',
            'Abstraction:',
            '```text',
            e['generated_principles_text'],
            '```',
            '',
        ])

    (final_root / 'REPORT_BREAKDOWN.md').write_text('\n'.join(report_lines), encoding='utf-8')
    (final_root / 'BEST_WORST_ABSTRACTIONS.md').write_text('\n'.join(examples_lines), encoding='utf-8')
    (final_root / 'summary_enriched.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps({
        'report': str(final_root / 'REPORT_BREAKDOWN.md'),
        'examples': str(final_root / 'BEST_WORST_ABSTRACTIONS.md'),
        'enriched_summary': str(final_root / 'summary_enriched.json'),
        'best_count': len(best),
        'worst_count': len(worst),
    }, indent=2))


if __name__ == '__main__':
    main()
