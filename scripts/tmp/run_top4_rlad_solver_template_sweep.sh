#!/bin/bash
set -euo pipefail

ROOT=/workspace/action_abstraction/outputs/2026-03-22/contrastive_abstraction_prompting/rlad_top4_rlad_solver_template_sweep
SAMPLE_DATASET=/workspace/action_abstraction/outputs/2026-03-21/contrastive_abstraction_prompting/rlad_sample12_qwen_baseline_vs_sft_sweep/sample_dataset
SAMPLE_MANIFEST=/workspace/action_abstraction/outputs/2026-03-21/contrastive_abstraction_prompting/rlad_sample12_qwen_baseline_vs_sft_sweep/sample_manifest.json
SOLVER_TEMPLATE=/workspace/action_abstraction/prompt_templates/rlad_solver_template.txt
ABSTRACTION_TEMPLATE=/workspace/action_abstraction/prompt_templates/sft_principle_generation.txt
RUN_ROOT=/workspace/action_abstraction/sft_models/Qwen3_1_7B-principle_generation_non_regressed_9k_subset_all_correct_v5_seed100_prompt_sft_principle_generation_half_epoch_eval/20260322_023906
FAMILY_SUMMARY=$ROOT/family_summary
FINAL_REPORT_ROOT=$ROOT/final_report
mkdir -p "$ROOT"

source /workspace/miniconda3/etc/profile.d/conda.sh
conda activate abstraction

echo "=== baseline with RLAD solver template ==="
/workspace/miniconda3/envs/abstraction/bin/python /workspace/action_abstraction/baseline_problem_solving.py \
  --input_dataset_path "$SAMPLE_DATASET" \
  --output_dir "$ROOT/baseline_solver_rlad_template" \
  --prompt_template_path "$SOLVER_TEMPLATE" \
  --cheatsheet_text "No cheatsheet provided." \
  --seed 100 \
  --temperature 0.6 \
  --top_p 0.95 \
  --top_k 20 \
  --n_solutions 4 \
  --max_tokens 32768

BASELINE_DATASET="$ROOT/baseline_solver_rlad_template/dataset"

for STEP in 1953 1302 1519 434; do
  CKPT="$RUN_ROOT/checkpoint-$STEP"
  SWEEP_ROOT="$ROOT/checkpoint_${STEP}_sweep"
  mkdir -p "$SWEEP_ROOT"
  echo "=== checkpoint-$STEP ==="
  for TEMP in 0.0 0.3 0.6 0.9; do
    SAFE=${TEMP/./p}
    echo "--- checkpoint-$STEP temp=$TEMP abstraction generation ---"
    /workspace/miniconda3/envs/abstraction/bin/python /tmp/run_rlad_abstraction_generation.py \
      --input_path "$BASELINE_DATASET" \
      --output_dir "$SWEEP_ROOT/abstractions_temp_${SAFE}" \
      --adapter_path "$CKPT" \
      --prompt_template_path "$ABSTRACTION_TEMPLATE" \
      --temperature "$TEMP" \
      --top_p 0.95 \
      --top_k 20 \
      --max_tokens 4096 \
      --n_abstractions 4 \
      --seed 100

    echo "--- checkpoint-$STEP temp=$TEMP conditioned solving (RLAD template) ---"
    /workspace/miniconda3/envs/abstraction/bin/python /workspace/action_abstraction/scripts/principle_conditioned_problem_solving.py \
      --input_path "$SWEEP_ROOT/abstractions_temp_${SAFE}/dataset" \
      --output_dir "$SWEEP_ROOT/conditioned_temp_${SAFE}" \
      --prompt_template_path "$SOLVER_TEMPLATE" \
      --principle_field generated_principles_text \
      --reference_passrate_field additional_passrate \
      --seed 100 \
      --temperature 0.6 \
      --top_p 0.95 \
      --top_k 20 \
      --n_solutions 4 \
      --max_tokens 32768
  done

  /workspace/miniconda3/envs/abstraction/bin/python /tmp/summarize_rlad_sweep.py \
    --sample_manifest "$SAMPLE_MANIFEST" \
    --baseline_dataset "$BASELINE_DATASET" \
    --conditioned_datasets \
      "$SWEEP_ROOT/conditioned_temp_0p0/dataset" \
      "$SWEEP_ROOT/conditioned_temp_0p3/dataset" \
      "$SWEEP_ROOT/conditioned_temp_0p6/dataset" \
      "$SWEEP_ROOT/conditioned_temp_0p9/dataset" \
    --output_dir "$SWEEP_ROOT/final_summary"

  /workspace/miniconda3/envs/abstraction/bin/python /tmp/make_rlad_report_docs.py \
    --root "$SWEEP_ROOT" \
    --label "checkpoint-$STEP"
done

/workspace/miniconda3/envs/abstraction/bin/python - <<'PY'
import json
from pathlib import Path
root = Path('/workspace/action_abstraction/outputs/2026-03-22/contrastive_abstraction_prompting/rlad_top4_rlad_solver_template_sweep')
steps = [1953, 1302, 1519, 434]
results = []
for step in steps:
    sweep_root = root / f'checkpoint_{step}_sweep'
    summary = json.loads((sweep_root / 'final_summary' / 'summary_enriched.json').read_text())
    best_avg = max(summary['temperatures'], key=lambda t: t['mean_conditioned_passrate'])
    best_best = max(summary['temperatures'], key=lambda t: t['mean_best_abstraction_passrate'])
    by_source = {str(t['temperature']): t.get('by_source', {}) for t in summary['temperatures']}
    temp_key = str(best_avg['temperature'])
    results.append({
        'checkpoint': f'checkpoint-{step}',
        'step': step,
        'best_average_temperature': best_avg['temperature'],
        'best_average_passrate': best_avg['mean_conditioned_passrate'],
        'best_best_of_4_temperature': best_best['temperature'],
        'best_best_of_4_passrate': best_best['mean_best_abstraction_passrate'],
        'aime_mean_at_best_average_temp': by_source.get(temp_key, {}).get('aime', {}).get('mean_conditioned_passrate'),
        'amc_mean_at_best_average_temp': by_source.get(temp_key, {}).get('amc', {}).get('mean_conditioned_passrate'),
        'report_path': str(sweep_root / 'final_summary' / 'REPORT_BREAKDOWN.md'),
        'examples_path': str(sweep_root / 'final_summary' / 'BEST_WORST_ABSTRACTIONS.md'),
        'summary_path': str(sweep_root / 'final_summary' / 'summary.json'),
    })
results.sort(key=lambda r: (-r['best_average_passrate'], r['step']))
baseline = json.loads((root / 'baseline_solver_rlad_template' / 'report.json').read_text())
payload = {
    'title': 'Top-4 checkpoint RLAD solver-template sweep',
    'selection_rule': 'top 4 checkpoints by prior best_average_passrate on the original RLAD sweep',
    'selected_checkpoints': [1953, 1302, 1519, 434],
    'solver_template': '/workspace/action_abstraction/prompt_templates/rlad_solver_template.txt',
    'abstraction_used_as': 'cheatsheet',
    'baseline_report': str(root / 'baseline_solver_rlad_template' / 'report.json'),
    'baseline_mean_passrate': baseline['mean_additional_passrate'],
    'checkpoints': results,
    'best_checkpoint_by_average': results[0] if results else None,
    'best_checkpoint_by_best_of_4': max(results, key=lambda r: r['best_best_of_4_passrate']) if results else None,
}
(root / 'family_summary').mkdir(parents=True, exist_ok=True)
(root / 'family_summary' / 'summary.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
lines = [
    '# Top-4 Checkpoint RLAD Solver-Template Sweep',
    '',
    f"- selection rule: `{payload['selection_rule']}`",
    f"- selected checkpoints: `{payload['selected_checkpoints']}`",
    f"- solver template: `{payload['solver_template']}`",
    f"- baseline mean passrate: `{payload['baseline_mean_passrate']:.4f}`",
    '',
    '## Checkpoint Table',
    '',
    '| checkpoint | best avg temp | best avg | AIME at best avg | AMC at best avg | best-of-4 temp | best-of-4 |',
    '| --- | ---: | ---: | ---: | ---: | ---: | ---: |',
]
for row in results:
    def f(x):
        return '' if x is None else f'{x:.4f}'
    lines.append(f"| {row['checkpoint']} | {row['best_average_temperature']} | {f(row['best_average_passrate'])} | {f(row['aime_mean_at_best_average_temp'])} | {f(row['amc_mean_at_best_average_temp'])} | {row['best_best_of_4_temperature']} | {f(row['best_best_of_4_passrate'])} |")
(root / 'family_summary' / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
print(json.dumps({'summary': str(root / 'family_summary' / 'summary.json'), 'report': str(root / 'family_summary' / 'REPORT.md')}, indent=2))
PY
