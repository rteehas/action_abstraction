#!/usr/bin/env bash
set -euo pipefail
source /workspace/miniconda3/etc/profile.d/conda.sh
conda activate abstraction
cd /workspace/action_abstraction

ROOT='outputs/2026-03-21/contrastive_abstraction_prompting/rlad_sample12_qwen_baseline_vs_sft_sweep'
BASELINE_DATASET="$ROOT/baseline_solver/dataset"
ADAPTER='sft_models/Qwen3_1_7B-principle_generation_improved_9k_subset_all_correct_v5_seed100_prompt_sft_principle_generation/20260321_185316/checkpoint-548'
PROMPT='prompt_templates/sft_principle_generation.txt'

for temp in 0.0 0.3 0.6 0.9; do
  safe=${temp/./p}
  abs_dir="$ROOT/abstractions_temp_${safe}"
  cond_dir="$ROOT/conditioned_temp_${safe}"

  if [ ! -d "$abs_dir/dataset" ]; then
    python /tmp/run_rlad_abstraction_generation.py \
      --input_path "$BASELINE_DATASET" \
      --output_dir "$abs_dir" \
      --adapter_path "$ADAPTER" \
      --prompt_template_path "$PROMPT" \
      --temperature "$temp" \
      --top_p 0.95 \
      --top_k 20 \
      --max_tokens 4096 \
      --n_abstractions 4 \
      --seed 100
  fi

  if [ ! -d "$cond_dir/dataset" ]; then
    python scripts/principle_conditioned_problem_solving.py \
      --input_path "$abs_dir/dataset" \
      --output_dir "$cond_dir" \
      --principle_field generated_principles_text \
      --reference_passrate_field additional_passrate \
      --seed 100 \
      --temperature 0.6 \
      --top_p 0.95 \
      --top_k 20 \
      --n_solutions 4 \
      --max_tokens 32768
  fi
done

python /tmp/summarize_rlad_sweep.py \
  --sample_manifest "$ROOT/sample_manifest.json" \
  --baseline_dataset "$BASELINE_DATASET" \
  --conditioned_datasets \
    "$ROOT/conditioned_temp_0p0/dataset" \
    "$ROOT/conditioned_temp_0p3/dataset" \
    "$ROOT/conditioned_temp_0p6/dataset" \
    "$ROOT/conditioned_temp_0p9/dataset" \
  --output_dir "$ROOT/final_summary"
