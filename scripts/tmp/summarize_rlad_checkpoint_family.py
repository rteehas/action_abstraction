from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_root', required=True)
    parser.add_argument('--sweeps_root', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--steps_per_epoch', type=float, required=True)
    args = parser.parse_args()

    run_root = Path(args.run_root)
    sweeps_root = Path(args.sweeps_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dirs = sorted([p for p in run_root.iterdir() if p.is_dir() and p.name.startswith('checkpoint-')], key=lambda p: int(p.name.split('-')[-1]))
    checkpoint_steps = [int(p.name.split('-')[-1]) for p in checkpoint_dirs]

    results = []
    for step in checkpoint_steps:
        sweep_root = sweeps_root / f'checkpoint_{step:04d}_sweep'
        summary_path = sweep_root / 'final_summary' / 'summary.json'
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text())
        temps = summary['temperatures']
        best_avg = max(temps, key=lambda t: t['mean_conditioned_passrate'])
        best_best = max(temps, key=lambda t: t['mean_best_abstraction_passrate'])

        checkpoint_path = run_root / f'checkpoint-{step}' / 'trainer_state.json'
        trainer_state = json.loads(checkpoint_path.read_text()) if checkpoint_path.exists() else {}
        last_eval = None
        if trainer_state.get('log_history'):
            eval_rows = [row for row in trainer_state['log_history'] if row.get('step') == step and 'eval_loss' in row]
            if eval_rows:
                last_eval = eval_rows[-1]

        rows_path = sweep_root / 'final_summary' / 'summary_enriched.json'
        enriched = json.loads(rows_path.read_text()) if rows_path.exists() else summary
        by_source = {str(t['temperature']): t.get('by_source', {}) for t in enriched['temperatures']}
        best_avg_temp = str(best_avg['temperature'])

        results.append({
            'checkpoint': f'checkpoint-{step}',
            'step': step,
            'epoch': step / args.steps_per_epoch,
            'trainer_eval_loss': None if last_eval is None else last_eval.get('eval_loss'),
            'trainer_eval_mean_token_accuracy': None if last_eval is None else last_eval.get('eval_mean_token_accuracy'),
            'best_average_temperature': best_avg['temperature'],
            'best_average_passrate': best_avg['mean_conditioned_passrate'],
            'best_average_improved_count': best_avg['improved_count'],
            'best_best_of_4_temperature': best_best['temperature'],
            'best_best_of_4_passrate': best_best['mean_best_abstraction_passrate'],
            'aime_mean_at_best_average_temp': by_source.get(best_avg_temp, {}).get('aime', {}).get('mean_conditioned_passrate'),
            'amc_mean_at_best_average_temp': by_source.get(best_avg_temp, {}).get('amc', {}).get('mean_conditioned_passrate'),
            'report_path': str(sweep_root / 'final_summary' / 'REPORT_BREAKDOWN.md'),
            'best_worst_path': str(sweep_root / 'final_summary' / 'BEST_WORST_ABSTRACTIONS.md'),
            'summary_path': str(summary_path),
        })

    results.sort(key=lambda row: row['step'])
    best_checkpoint = max(results, key=lambda row: row['best_average_passrate']) if results else None
    best_best_checkpoint = max(results, key=lambda row: row['best_best_of_4_passrate']) if results else None

    payload = {
        'run_root': str(run_root),
        'sweeps_root': str(sweeps_root),
        'steps_per_epoch': args.steps_per_epoch,
        'num_checkpoints_evaluated': len(results),
        'best_checkpoint_by_average': best_checkpoint,
        'best_checkpoint_by_best_of_4': best_best_checkpoint,
        'checkpoints': results,
    }
    (output_dir / 'summary.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')

    lines = [
        '# RLAD Checkpoint Family Summary',
        '',
        f"- run root: `{run_root}`",
        f"- sweeps root: `{sweeps_root}`",
        f"- checkpoints evaluated: `{len(results)}`",
        '',
    ]
    if best_checkpoint is not None:
        lines.extend([
            '## Best Checkpoints',
            '',
            f"- best average passrate: `{best_checkpoint['checkpoint']}` at epoch `{best_checkpoint['epoch']:.2f}` with `{best_checkpoint['best_average_passrate']:.4f}` at temp `{best_checkpoint['best_average_temperature']}`",
            f"- best best-of-4 passrate: `{best_best_checkpoint['checkpoint']}` at epoch `{best_best_checkpoint['epoch']:.2f}` with `{best_best_checkpoint['best_best_of_4_passrate']:.4f}` at temp `{best_best_checkpoint['best_best_of_4_temperature']}`",
            '',
            '## Checkpoint Table',
            '',
            '| checkpoint | epoch | trainer eval loss | trainer eval token acc | best avg temp | best avg | AIME at best avg | AMC at best avg | best-of-4 temp | best-of-4 |',
            '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
        ])
        for row in results:
            def f(x):
                return '' if x is None else f'{x:.4f}'
            lines.append(
                f"| {row['checkpoint']} | {row['epoch']:.2f} | {f(row['trainer_eval_loss'])} | {f(row['trainer_eval_mean_token_accuracy'])} | {row['best_average_temperature']} | {f(row['best_average_passrate'])} | {f(row['aime_mean_at_best_average_temp'])} | {f(row['amc_mean_at_best_average_temp'])} | {row['best_best_of_4_temperature']} | {f(row['best_best_of_4_passrate'])} |"
            )
    (output_dir / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({'summary': str(output_dir / 'summary.json'), 'report': str(output_dir / 'REPORT.md')}, indent=2))


if __name__ == '__main__':
    main()
