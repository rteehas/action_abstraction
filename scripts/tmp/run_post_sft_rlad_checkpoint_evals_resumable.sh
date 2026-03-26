#!/bin/bash
set -euo pipefail

TRAIN_PID=${TRAIN_PID:-416187}
RUN_ROOT=${RUN_ROOT:-/workspace/action_abstraction/sft_models/Qwen3_1_7B-principle_generation_non_regressed_9k_subset_all_correct_v5_seed100_prompt_sft_principle_generation_half_epoch_eval/20260322_023906}
BASE_ROOT=${BASE_ROOT:-/workspace/action_abstraction/outputs/2026-03-21/contrastive_abstraction_prompting/rlad_sample12_qwen_baseline_vs_sft_sweep}
SAMPLE_MANIFEST=${SAMPLE_MANIFEST:-$BASE_ROOT/sample_manifest.json}
BASELINE_DATASET=${BASELINE_DATASET:-$BASE_ROOT/baseline_solver/dataset}
PROMPT_TEMPLATE=${PROMPT_TEMPLATE:-/workspace/action_abstraction/prompt_templates/sft_principle_generation.txt}
SWEEPS_ROOT=${SWEEPS_ROOT:-/workspace/action_abstraction/outputs/2026-03-22/contrastive_abstraction_prompting/rlad_non_regressed_half_epoch_checkpoint_sweeps}
FAMILY_SUMMARY_ROOT=${FAMILY_SUMMARY_ROOT:-$SWEEPS_ROOT/family_summary}
STEPS_PER_EPOCH=${STEPS_PER_EPOCH:-434}
mkdir -p "$SWEEPS_ROOT"

source /workspace/miniconda3/etc/profile.d/conda.sh
conda activate abstraction

echo "[resumable] waiting for training pid $TRAIN_PID"
while kill -0 "$TRAIN_PID" 2>/dev/null; do
  date -u +"%Y-%m-%dT%H:%M:%SZ [resumable] training still running"
  sleep 60
done

date -u +"%Y-%m-%dT%H:%M:%SZ [resumable] training finished or exited"

mapfile -t CHECKPOINTS < <(find "$RUN_ROOT" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V)
if [ ${#CHECKPOINTS[@]} -eq 0 ]; then
  echo "[resumable] no checkpoints found under $RUN_ROOT" >&2
  exit 1
fi

for CKPT in "${CHECKPOINTS[@]}"; do
  STEP=$(basename "$CKPT" | sed 's/^checkpoint-//')
  ROOT="$SWEEPS_ROOT/checkpoint_$(printf '%04d' "$STEP")_sweep"
  FINAL_SUMMARY="$ROOT/final_summary/summary.json"
  mkdir -p "$ROOT"

  if [ -f "$FINAL_SUMMARY" ]; then
    echo "[resumable] checkpoint-$STEP already summarized; skipping"
    continue
  fi

  echo "[resumable] === evaluating checkpoint-$STEP ==="
  for TEMP in 0.0 0.3 0.6 0.9; do
    SAFE=${TEMP/./p}
    ABS_DATASET="$ROOT/abstractions_temp_${SAFE}/dataset"
    COND_DATASET="$ROOT/conditioned_temp_${SAFE}/dataset"

    if [ ! -d "$ABS_DATASET" ]; then
      echo "[resumable] --- checkpoint-$STEP temp=$TEMP abstraction generation ---"
      /workspace/miniconda3/envs/abstraction/bin/python /tmp/run_rlad_abstraction_generation.py \
        --input_path "$BASELINE_DATASET" \
        --output_dir "$ROOT/abstractions_temp_${SAFE}" \
        --adapter_path "$CKPT" \
        --prompt_template_path "$PROMPT_TEMPLATE" \
        --temperature "$TEMP" \
        --top_p 0.95 \
        --top_k 20 \
        --max_tokens 4096 \
        --n_abstractions 4 \
        --seed 100
    else
      echo "[resumable] --- checkpoint-$STEP temp=$TEMP abstraction generation already done ---"
    fi

    if [ ! -d "$COND_DATASET" ]; then
      echo "[resumable] --- checkpoint-$STEP temp=$TEMP conditioned solving ---"
      /workspace/miniconda3/envs/abstraction/bin/python /workspace/action_abstraction/scripts/principle_conditioned_problem_solving.py \
        --input_path "$ROOT/abstractions_temp_${SAFE}/dataset" \
        --output_dir "$ROOT/conditioned_temp_${SAFE}" \
        --principle_field generated_principles_text \
        --reference_passrate_field additional_passrate \
        --seed 100 \
        --temperature 0.6 \
        --top_p 0.95 \
        --top_k 20 \
        --n_solutions 4 \
        --max_tokens 32768
    else
      echo "[resumable] --- checkpoint-$STEP temp=$TEMP conditioned solving already done ---"
    fi
  done

  /workspace/miniconda3/envs/abstraction/bin/python /tmp/summarize_rlad_sweep.py \
    --sample_manifest "$SAMPLE_MANIFEST" \
    --baseline_dataset "$BASELINE_DATASET" \
    --conditioned_datasets \
      "$ROOT/conditioned_temp_0p0/dataset" \
      "$ROOT/conditioned_temp_0p3/dataset" \
      "$ROOT/conditioned_temp_0p6/dataset" \
      "$ROOT/conditioned_temp_0p9/dataset" \
    --output_dir "$ROOT/final_summary"

  /workspace/miniconda3/envs/abstraction/bin/python /tmp/make_rlad_report_docs.py \
    --root "$ROOT" \
    --label "checkpoint-$STEP"
done

/workspace/miniconda3/envs/abstraction/bin/python /tmp/summarize_rlad_checkpoint_family.py \
  --run_root "$RUN_ROOT" \
  --sweeps_root "$SWEEPS_ROOT" \
  --output_dir "$FAMILY_SUMMARY_ROOT" \
  --steps_per_epoch "$STEPS_PER_EPOCH"
