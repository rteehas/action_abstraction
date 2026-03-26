#!/bin/bash
set -euo pipefail
cd /workspace/action_abstraction
export VLLM_USE_FLASHINFER_SAMPLER=0
OUT_ROOT="outputs/2026-03-23/contrastive_abstraction_prompting/rlad_hint_gen_merged_temp0p6_qwen_solver_temp0p6_sample12_native_sampler"
INPUT="outputs/2026-03-21/contrastive_abstraction_prompting/rlad_sample12_qwen_baseline_vs_sft_sweep/sample_dataset"
mkdir -p "$OUT_ROOT" "$OUT_ROOT/abstractions_temp_0p6" "$OUT_ROOT/conditioned_temp_0p6"
/workspace/miniconda3/envs/abstraction/bin/python /tmp/run_rlad_abstraction_generation_merged.py \
  --input_path "$INPUT" \
  --output_dir "$OUT_ROOT/abstractions_temp_0p6" \
  --model CMU-AIRe/RLAD-Hint-Gen \
  --prompt_template_path /workspace/action_abstraction/prompt_templates/rlad_abstraction_generation_prompt_template.txt \
  --temperature 0.6 \
  --top_p 0.95 \
  --top_k 20 \
  --max_tokens 4096 \
  --n_abstractions 4 \
  --seed 100 \
  --tensor_parallel_size 1 \
  --gpu_memory_utilization 0.9 \
  --max_num_batched_tokens 4096 | tee "$OUT_ROOT/abstractions_temp_0p6/run.log"
/workspace/miniconda3/envs/abstraction/bin/python scripts/principle_conditioned_problem_solving.py \
  --input_path "$OUT_ROOT/abstractions_temp_0p6/dataset" \
  --output_dir "$OUT_ROOT/conditioned_temp_0p6" \
  --base_model Qwen/Qwen3-1.7B \
  --prompt_template_path /workspace/action_abstraction/prompt_templates/rlad_solver_template.txt \
  --principle_field generated_principles_text \
  --reference_passrate_field additional_passrate \
  --result_prefix conditioned \
  --seed 100 \
  --temperature 0.6 \
  --top_p 0.95 \
  --top_k 20 \
  --n_solutions 4 \
  --max_tokens 32768 \
  --tensor_parallel_size 1 \
  --gpu_memory_utilization 0.9 \
  --max_num_batched_tokens 4096 | tee "$OUT_ROOT/conditioned_temp_0p6/run.log"
