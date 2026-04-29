#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# Throughput-oriented wrapper for the sample128 two-policy debug run.
# This keeps the original debug launcher intact while opting into denser
# rollout/training settings when the current node has memory headroom.
export RUN_TAG=${RUN_TAG:-sample128_dense_bs16_abs4_sol8_$(date -u +%Y%m%d_%H%M%S)}
export TRAINER_EXPERIMENT_NAME=${TRAINER_EXPERIMENT_NAME:-deepscaler_sample128_dense_qwen3_1p7b}

export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-16}
export VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-8}
export ACTOR_PPO_MINI_BATCH_SIZE=${ACTOR_PPO_MINI_BATCH_SIZE:-8}
export ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU=${ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU:-24576}

export SOLVER_GPU_MEM_UTIL=${SOLVER_GPU_MEM_UTIL:-0.42}
export ABSTRACTION_GPU_MEM_UTIL=${ABSTRACTION_GPU_MEM_UTIL:-0.22}
export ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-64}
export SOLVER_MAX_BATCHED_TOKENS=${SOLVER_MAX_BATCHED_TOKENS:-8192}
export ABSTRACTION_MAX_BATCHED_TOKENS=${ABSTRACTION_MAX_BATCHED_TOKENS:-2048}

export ROLLOUT_ENFORCE_EAGER=${ROLLOUT_ENFORCE_EAGER:-False}
export ROLLOUT_FREE_CACHE_ENGINE=${ROLLOUT_FREE_CACHE_ENGINE:-False}
export ACTOR_PARAM_OFFLOAD=${ACTOR_PARAM_OFFLOAD:-False}
export ACTOR_OPTIMIZER_OFFLOAD=${ACTOR_OPTIMIZER_OFFLOAD:-False}

export RAY_NUM_CPUS=${RAY_NUM_CPUS:-4}
export DATA_DATALOADER_NUM_WORKERS=${DATA_DATALOADER_NUM_WORKERS:-1}
export REWARD_NUM_WORKERS=${REWARD_NUM_WORKERS:-1}
export SOLVER_AGENT_NUM_WORKERS=${SOLVER_AGENT_NUM_WORKERS:-1}
export ABSTRACTION_AGENT_NUM_WORKERS=${ABSTRACTION_AGENT_NUM_WORKERS:-1}

echo "Sample128 throughput launch:"
echo "  train_batch_size=${TRAIN_BATCH_SIZE}"
echo "  ppo_mini_batch_size=${ACTOR_PPO_MINI_BATCH_SIZE}"
echo "  solver_gpu_mem_util=${SOLVER_GPU_MEM_UTIL}"
echo "  abstraction_gpu_mem_util=${ABSTRACTION_GPU_MEM_UTIL}"
echo "  rollout_max_num_seqs=${ROLLOUT_MAX_NUM_SEQS}"
echo "  solver_max_batched_tokens=${SOLVER_MAX_BATCHED_TOKENS}"
echo "  abstraction_max_batched_tokens=${ABSTRACTION_MAX_BATCHED_TOKENS}"
echo "  rollout_enforce_eager=${ROLLOUT_ENFORCE_EAGER}"
echo "  rollout_free_cache_engine=${ROLLOUT_FREE_CACHE_ENGINE}"
echo "  actor_param_offload=${ACTOR_PARAM_OFFLOAD}"
echo "  actor_optimizer_offload=${ACTOR_OPTIMIZER_OFFLOAD}"
echo "  data_dataloader_num_workers=${DATA_DATALOADER_NUM_WORKERS}"
echo "  reward_num_workers=${REWARD_NUM_WORKERS}"
echo "  solver_agent_num_workers=${SOLVER_AGENT_NUM_WORKERS}"
echo "  abstraction_agent_num_workers=${ABSTRACTION_AGENT_NUM_WORKERS}"

exec bash "${SCRIPT_DIR}/run_qwen_two_policy_deepscaler_sample128_debug.sh" "$@"
