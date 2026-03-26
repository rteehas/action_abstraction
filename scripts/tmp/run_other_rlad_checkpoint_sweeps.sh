#!/bin/bash
set -euo pipefail

source /workspace/miniconda3/etc/profile.d/conda.sh
conda activate abstraction

BASE_ROOT=/workspace/action_abstraction/outputs/2026-03-21/contrastive_abstraction_prompting/rlad_sample12_qwen_baseline_vs_sft_sweep
SAMPLE_MANIFEST="$BASE_ROOT/sample_manifest.json"
BASELINE_DATASET="$BASE_ROOT/baseline_solver/dataset"
PROMPT_TEMPLATE=/workspace/action_abstraction/prompt_templates/sft_principle_generation.txt
MODEL_ROOT=/workspace/action_abstraction/sft_models/Qwen3_1_7B-principle_generation_improved_9k_subset_all_correct_v5_seed100_prompt_sft_principle_generation/20260321_185316
OUT_BASE=/workspace/action_abstraction/outputs/2026-03-21/contrastive_abstraction_prompting

for CKPT in 137 274 411; do
  ROOT="$OUT_BASE/rlad_sample12_qwen_checkpoint_${CKPT}_sweep"
  ADAPTER="$MODEL_ROOT/checkpoint-$CKPT"
  mkdir -p "$ROOT"
  echo "=== checkpoint-$CKPT ==="
  for TEMP in 0.0 0.3 0.6 0.9; do
    SAFE=${TEMP/./p}
    echo "--- checkpoint-$CKPT temp=$TEMP abstraction generation ---"
    /workspace/miniconda3/envs/abstraction/bin/python /tmp/run_rlad_abstraction_generation.py \
      --input_path "$BASELINE_DATASET" \
      --output_dir "$ROOT/abstractions_temp_${SAFE}" \
      --adapter_path "$ADAPTER" \
      --prompt_template_path "$PROMPT_TEMPLATE" \
      --temperature "$TEMP" \
      --top_p 0.95 \
      --top_k 20 \
      --max_tokens 4096 \
      --n_abstractions 4 \
      --seed 100

    echo "--- checkpoint-$CKPT temp=$TEMP conditioned solving ---"
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
    --label "checkpoint-$CKPT"
done
