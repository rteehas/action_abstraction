#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# Sample128 debug launch with the H200-balanced rollout settings from
# run_qwen_two_policy_full_dapo.sh, while keeping the debug script's
# smaller batch shape and disabled validation cadence.
export RUN_TAG=${RUN_TAG:-sample128_h200_balanced_bs8_abs4_sol8_$(date -u +%Y%m%d_%H%M%S)}
export TRAINER_EXPERIMENT_NAME=${TRAINER_EXPERIMENT_NAME:-deepscaler_sample128_h200_balanced_qwen3_1p7b}

export GPU_TUNING_PRESET=${GPU_TUNING_PRESET:-h200_balanced}
export SOLVER_GPU_MEM_UTIL=${SOLVER_GPU_MEM_UTIL:-0.50}
export ABSTRACTION_GPU_MEM_UTIL=${ABSTRACTION_GPU_MEM_UTIL:-0.28}
export ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-144}
export SOLVER_MAX_BATCHED_TOKENS=${SOLVER_MAX_BATCHED_TOKENS:-10240}
export ABSTRACTION_MAX_BATCHED_TOKENS=${ABSTRACTION_MAX_BATCHED_TOKENS:-2560}

export ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU=${ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU:-32768}
export RAY_NUM_CPUS=${RAY_NUM_CPUS:-8}

echo "Sample128 H200-balanced launch:"
echo "  gpu_tuning_preset=${GPU_TUNING_PRESET}"
echo "  solver_gpu_mem_util=${SOLVER_GPU_MEM_UTIL}"
echo "  abstraction_gpu_mem_util=${ABSTRACTION_GPU_MEM_UTIL}"
echo "  rollout_max_num_seqs=${ROLLOUT_MAX_NUM_SEQS}"
echo "  solver_max_batched_tokens=${SOLVER_MAX_BATCHED_TOKENS}"
echo "  abstraction_max_batched_tokens=${ABSTRACTION_MAX_BATCHED_TOKENS}"
echo "  actor_ppo_max_token_len_per_gpu=${ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU}"
echo "  ray_num_cpus=${RAY_NUM_CPUS}"

exec bash "${SCRIPT_DIR}/run_qwen_two_policy_deepscaler_sample128_debug.sh" "$@"
