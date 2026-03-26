from __future__ import annotations

import argparse
import json
from pathlib import Path


def write_plots(root: Path, family: dict) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return []
    checkpoints = family.get('checkpoints', [])
    if not checkpoints:
        return []
    order = [1953, 1302, 1519, 434]
    checkpoints = sorted(checkpoints, key=lambda r: order.index(r['step']) if r['step'] in order else r['step'])
    steps = [r['step'] for r in checkpoints]
    avg = [r['best_average_passrate'] for r in checkpoints]
    best4 = [r['best_best_of_4_passrate'] for r in checkpoints]
    aime = [r['aime_mean_at_best_average_temp'] for r in checkpoints]
    amc = [r['amc_mean_at_best_average_temp'] for r in checkpoints]
    plot_paths = []

    plt.figure(figsize=(8,5))
    plt.plot(steps, avg, marker='o', label='best average')
    plt.plot(steps, best4, marker='o', label='best-of-4')
    plt.xlabel('Checkpoint step')
    plt.ylabel('Passrate')
    plt.title('Top-4 rerun passrates with RLAD solver template')
    plt.grid(alpha=0.3)
    plt.legend()
    p = root / 'top4_passrate_curves.png'
    plt.tight_layout(); plt.savefig(p, dpi=160); plt.close()
    plot_paths.append(str(p))

    plt.figure(figsize=(8,5))
    plt.plot(steps, aime, marker='o', label='AIME at best avg temp')
    plt.plot(steps, amc, marker='o', label='AMC at best avg temp')
    plt.xlabel('Checkpoint step')
    plt.ylabel('Passrate')
    plt.title('Top-4 rerun AIME vs AMC')
    plt.grid(alpha=0.3)
    plt.legend()
    p = root / 'top4_aime_amc_curves.png'
    plt.tight_layout(); plt.savefig(p, dpi=160); plt.close()
    plot_paths.append(str(p))

    return plot_paths


def write_source_breakdown(root: Path) -> str:
    AIME_BASE = 0.375
    AMC_BASE = 0.875
    rows = []
    for p in sorted(root.glob('checkpoint_*_sweep/final_summary/summary_enriched.json')):
        summary = json.loads(p.read_text())
        step = int(p.parts[-3].split('_')[1])
        for t in summary['temperatures']:
            rows.append({
                'step': step,
                'temperature': t['temperature'],
                'overall': t['mean_conditioned_passrate'],
                'aime': t['by_source']['aime']['mean_conditioned_passrate'],
                'aime_best4': t['by_source']['aime']['mean_best_abstraction_passrate'],
                'amc': t['by_source']['amc']['mean_conditioned_passrate'],
                'amc_best4': t['by_source']['amc']['mean_best_abstraction_passrate'],
                'aime_counts': t['by_source']['aime'],
                'amc_counts': t['by_source']['amc'],
            })
    out = root / 'final_report' / 'CHECKPOINT_SOURCE_BREAKDOWN.md'
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        '# Checkpoint Source Breakdown',
        '',
        '- overall baseline: `0.6250`',
        f'- AIME baseline: `{AIME_BASE:.4f}`',
        f'- AMC baseline: `{AMC_BASE:.4f}`',
        '',
    ]
    for step in [1953, 1302, 1519, 434]:
        group = [r for r in rows if r['step'] == step]
        lines.extend([
            f'## checkpoint-{step}',
            '',
            '| temp | overall | AIME | Δ AIME | AIME best-of-4 | AIME I/S/R | AMC | Δ AMC | AMC best-of-4 | AMC I/S/R |',
            '| ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |',
        ])
        for r in sorted(group, key=lambda x: x['temperature']):
            a = r['aime_counts']
            m = r['amc_counts']
            lines.append(
                f"| {r['temperature']:.1f} | {r['overall']:.4f} | {r['aime']:.4f} | {r['aime']-AIME_BASE:+.4f} | {r['aime_best4']:.4f} | {a['improved_count']}/{a['same_count']}/{a['regressed_count']} | {r['amc']:.4f} | {r['amc']-AMC_BASE:+.4f} | {r['amc_best4']:.4f} | {m['improved_count']}/{m['same_count']}/{m['regressed_count']} |"
            )
        lines.append('')
    out.write_text('\n'.join(lines), encoding='utf-8')
    return str(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    args = parser.parse_args()
    root = Path(args.root)
    family_summary_path = root / 'family_summary' / 'summary.json'
    baseline_report_path = root / 'baseline_solver_rlad_template' / 'report.json'
    if not family_summary_path.exists():
        raise SystemExit(f'missing family summary: {family_summary_path}')
    family = json.loads(family_summary_path.read_text())
    baseline = json.loads(baseline_report_path.read_text())

    final_root = root / 'final_report'
    final_root.mkdir(parents=True, exist_ok=True)
    plot_paths = write_plots(final_root, family)
    source_breakdown_path = write_source_breakdown(root)

    best_avg = family.get('best_checkpoint_by_average')
    best_best = family.get('best_checkpoint_by_best_of_4')
    payload = {
        'root': str(root),
        'selection_rule': family.get('selection_rule'),
        'selected_checkpoints': family.get('selected_checkpoints'),
        'solver_template': family.get('solver_template'),
        'baseline_mean_passrate': family.get('baseline_mean_passrate'),
        'best_checkpoint_by_average': best_avg,
        'best_checkpoint_by_best_of_4': best_best,
        'baseline_report': str(baseline_report_path),
        'plots': plot_paths,
        'source_breakdown': source_breakdown_path,
    }
    (final_root / 'summary.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')

    lines = [
        '# Worklog 2026-03-22: RLAD Solver-Template Top-4 Rerun',
        '',
        '## Setup',
        '',
        f"- selection rule: `{family.get('selection_rule')}`",
        f"- selected checkpoints: `{family.get('selected_checkpoints')}`",
        f"- solver template: `{family.get('solver_template')}`",
        '- abstraction inserted into the template as the cheatsheet',
        '- baseline rerun used the same solver template with cheatsheet text `No cheatsheet provided.`',
        '- template fill was adjusted via direct string replacement for `{cheatsheet}` and `{problem_description}` rather than Python formatting, because the template also contains literal braces in `\\boxed{<answer>}`',
        '',
        '## Baseline',
        '',
        f"- baseline mean passrate under the RLAD solver template: `{family.get('baseline_mean_passrate'):.4f}`",
        f"- baseline report: `{baseline_report_path}`",
        '',
        '## Best Checkpoints',
        '',
        f"- best average checkpoint: `{best_avg['checkpoint']}` with `{best_avg['best_average_passrate']:.4f}` at temp `{best_avg['best_average_temperature']}`",
        f"- best best-of-4 checkpoint: `{best_best['checkpoint']}` with `{best_best['best_best_of_4_passrate']:.4f}` at temp `{best_best['best_best_of_4_temperature']}`",
        '',
        '## Checkpoint Table',
        '',
        '| checkpoint | best avg temp | best avg | AIME at best avg | AMC at best avg | best-of-4 temp | best-of-4 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for row in family.get('checkpoints', []):
        def f(x):
            return '' if x is None else f'{x:.4f}'
        lines.append(f"| {row['checkpoint']} | {row['best_average_temperature']} | {f(row['best_average_passrate'])} | {f(row['aime_mean_at_best_average_temp'])} | {f(row['amc_mean_at_best_average_temp'])} | {row['best_best_of_4_temperature']} | {f(row['best_best_of_4_passrate'])} |")
    lines.extend([
        '',
        '## Artifacts',
        '',
        f"- family summary: `{family_summary_path}`",
        f"- family report: `{root / 'family_summary' / 'REPORT.md'}`",
        f"- source breakdown: `{source_breakdown_path}`",
        f"- plots: `{plot_paths}`",
    ])
    text = '\n'.join(lines)
    (final_root / 'REPORT.md').write_text(text, encoding='utf-8')
    (final_root / 'WORKLOG_20260322.md').write_text(text, encoding='utf-8')
    print(json.dumps({'report': str(final_root / 'REPORT.md'), 'worklog': str(final_root / 'WORKLOG_20260322.md'), 'summary': str(final_root / 'summary.json')}, indent=2))


if __name__ == '__main__':
    main()
