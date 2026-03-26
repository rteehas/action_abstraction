from __future__ import annotations

import json
from pathlib import Path

BASE = Path('/workspace/action_abstraction/outputs/2026-03-21/contrastive_abstraction_prompting')
OUT_DIR = Path('/workspace/action_abstraction/outputs/2026-03-22/contrastive_abstraction_prompting/rlad_checkpoint_sweep_comparison')
OUT_DIR.mkdir(parents=True, exist_ok=True)

checkpoints = [
    (
        'checkpoint-137',
        BASE / 'rlad_sample12_qwen_checkpoint_137_sweep' / 'final_summary' / 'rows.json',
        BASE / 'rlad_sample12_qwen_checkpoint_137_sweep' / 'final_summary' / 'summary.json',
    ),
    (
        'checkpoint-274',
        BASE / 'rlad_sample12_qwen_checkpoint_274_sweep' / 'final_summary' / 'rows.json',
        BASE / 'rlad_sample12_qwen_checkpoint_274_sweep' / 'final_summary' / 'summary.json',
    ),
    (
        'checkpoint-411',
        BASE / 'rlad_sample12_qwen_checkpoint_411_sweep' / 'final_summary' / 'rows.json',
        BASE / 'rlad_sample12_qwen_checkpoint_411_sweep' / 'final_summary' / 'summary.json',
    ),
    (
        'checkpoint-548-final',
        BASE / 'rlad_sample12_qwen_baseline_vs_sft_sweep' / 'final_summary' / 'rows.json',
        BASE / 'rlad_sample12_qwen_baseline_vs_sft_sweep' / 'final_summary' / 'summary.json',
    ),
]

all_rows: dict[str, list[dict]] = {}
all_summaries: dict[str, dict] = {}
for label, rows_path, summary_path in checkpoints:
    all_rows[label] = json.loads(rows_path.read_text())
    all_summaries[label] = json.loads(summary_path.read_text())

canonical = all_rows['checkpoint-548-final']
rows_by_ckpt = {
    label: {row['sample_idx']: row for row in rows}
    for label, rows in all_rows.items()
}

selected_temps = {
    label: [str(t['temperature']) for t in summary['temperatures']]
    for label, summary in all_summaries.items()
}

def fmt(x: float) -> str:
    return f"{x:.4f}".rstrip('0').rstrip('.') if isinstance(x, float) else str(x)

lines: list[str] = []
lines.extend([
    '# RLAD Checkpoint Abstraction Comparison',
    '',
    '- dataset: `/workspace/rlad_aime_amc_scored`',
    '- sample: 12 problems, balanced `6` AIME + `6` AMC, fixed seed `100`',
    '- baseline solver: `Qwen/Qwen3-1.7B`, `4` samples, thinking on',
    '- conditioned solver: `Qwen/Qwen3-1.7B`, `4` samples, thinking on',
    '- abstraction generator temperatures shown for every checkpoint: `0.0`, `0.3`, `0.6`, `0.9`',
    '- note: for abstraction temperature `0.0`, the generator produces one greedy abstraction and the sweep duplicates it into four identical abstraction slots',
    '',
    'This note is organized as: problem -> checkpoint -> temperature -> abstraction.',
    'For each abstraction, `delta` means `abstraction-conditioned passrate - baseline passrate` for that same problem.',
    '',
])

for row in canonical:
    sample_idx = row['sample_idx']
    lines.extend([
        f"## Problem {sample_idx}",
        '',
        f"- dataset row: `{row['dataset_idx']}`",
        f"- source: `{row['source']}`",
        f"- answer: `{row['answer']}`",
        f"- baseline passrate: `{fmt(float(row['baseline_passrate']))}`",
        f"- baseline generated answers: `{row.get('baseline_generated_answer')}`",
        '',
        'Problem:',
        '```text',
        row['problem'],
        '```',
        '',
    ])
    for label, _, _ in checkpoints:
        ckpt_row = rows_by_ckpt[label][sample_idx]
        lines.extend([
            f"### {label}",
            '',
        ])
        for temp in selected_temps[label]:
            temp_result = ckpt_row['temperature_results'][temp]
            agg = float(temp_result['aggregate_conditioned_passrate'])
            best = float(temp_result['best_abstraction_passrate'])
            delta_agg = agg - float(ckpt_row['baseline_passrate'])
            delta_best = best - float(ckpt_row['baseline_passrate'])
            lines.extend([
                f"#### temperature = {temp}",
                '',
                f"- aggregate conditioned passrate: `{fmt(agg)}`",
                f"- aggregate delta vs baseline: `{delta_agg:+.4f}`",
                f"- best abstraction passrate: `{fmt(best)}`",
                f"- best delta vs baseline: `{delta_best:+.4f}`",
                '',
            ])
            abstractions = sorted(temp_result['abstractions'], key=lambda x: x['abstraction_idx'])
            for abstraction in abstractions:
                p = float(abstraction['conditioned_passrate'])
                delta = p - float(ckpt_row['baseline_passrate'])
                lines.extend([
                    f"1. Abstraction `{abstraction['abstraction_idx']}`: passrate `{fmt(p)}`, delta `{delta:+.4f}`",
                    f"Conditioned answers: `{abstraction.get('conditioned_generated_answer')}`",
                    '```text',
                    abstraction['generated_principles_text'],
                    '```',
                    '',
                ])
        lines.append('')

out_path = OUT_DIR / 'CHECKPOINT_ABSTRACTIONS_BY_PROBLEM.md'
out_path.write_text('\n'.join(lines), encoding='utf-8')

summary = {
    'output_path': str(out_path),
    'num_problems': len(canonical),
    'checkpoints': [label for label, _, _ in checkpoints],
    'temperatures': selected_temps,
}
(OUT_DIR / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
print(json.dumps(summary, indent=2))
