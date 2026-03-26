#!/bin/bash
set -euo pipefail

TRAIN_PID=${TRAIN_PID:-416187}
BATCH_PID=${BATCH_PID:-416955}
RUN_ROOT=${RUN_ROOT:-/workspace/action_abstraction/sft_models/Qwen3_1_7B-principle_generation_non_regressed_9k_subset_all_correct_v5_seed100_prompt_sft_principle_generation_half_epoch_eval/20260322_023906}
DATASET_PATH=${DATASET_PATH:-/workspace/action_abstraction/sft_datasets/principle_generation_non_regressed_9k_subset_all_correct_v5_seed100}
SWEEPS_ROOT=${SWEEPS_ROOT:-/workspace/action_abstraction/outputs/2026-03-22/contrastive_abstraction_prompting/rlad_non_regressed_half_epoch_checkpoint_sweeps}
FAMILY_SUMMARY=${FAMILY_SUMMARY:-$SWEEPS_ROOT/family_summary/summary.json}
BASELINE_SUMMARY=${BASELINE_SUMMARY:-/workspace/action_abstraction/outputs/2026-03-21/contrastive_abstraction_prompting/rlad_sample12_qwen_baseline_vs_sft_sweep/final_summary/summary.json}
REPORT_ROOT=${REPORT_ROOT:-/workspace/action_abstraction/outputs/2026-03-22/contrastive_abstraction_prompting/sft_principle_generation_non_regressed_9k_subset_all_correct_v5_seed100_half_epoch_eval_report}
RESUME_LOG=${RESUME_LOG:-/workspace/action_abstraction/outputs/2026-03-22/contrastive_abstraction_prompting/rlad_non_regressed_half_epoch_checkpoint_sweeps.resume.log}

source /workspace/miniconda3/etc/profile.d/conda.sh
conda activate abstraction

echo "[watcher] waiting for batch pid $BATCH_PID"
while kill -0 "$BATCH_PID" 2>/dev/null; do
  date -u +"%Y-%m-%dT%H:%M:%SZ [watcher] batch still running"
  sleep 120
done

date -u +"%Y-%m-%dT%H:%M:%SZ [watcher] batch exited"

if [ ! -f "$FAMILY_SUMMARY" ]; then
  echo "[watcher] family summary missing; starting resumable sweep" | tee -a "$RESUME_LOG"
  TRAIN_PID="$TRAIN_PID" RUN_ROOT="$RUN_ROOT" SWEEPS_ROOT="$SWEEPS_ROOT" \
    /tmp/run_post_sft_rlad_checkpoint_evals_resumable.sh >> "$RESUME_LOG" 2>&1
else
  echo "[watcher] family summary already present; no resumable sweep needed" | tee -a "$RESUME_LOG"
fi

if [ ! -f "$FAMILY_SUMMARY" ]; then
  echo "[watcher] family summary still missing after resume attempt" | tee -a "$RESUME_LOG"
  exit 1
fi

/workspace/miniconda3/envs/abstraction/bin/python /tmp/generate_non_regressed_sft_rlad_report.py \
  --dataset_path "$DATASET_PATH" \
  --run_root "$RUN_ROOT" \
  --family_summary "$FAMILY_SUMMARY" \
  --baseline_summary "$BASELINE_SUMMARY" \
  --output_dir "$REPORT_ROOT" \
  --sweeps_root "$SWEEPS_ROOT" >> "$RESUME_LOG" 2>&1

echo "[watcher] final report generated under $REPORT_ROOT" | tee -a "$RESUME_LOG"
