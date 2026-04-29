#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}

export REPO_ROOT

DATASET_DIR=${DATASET_DIR:-${REPO_ROOT}/verl_data/two_policy_deepscaler_qwne1_7b_passrate_025_075_sample128_seed0}
TRAIN_FILES=${TRAIN_FILES:-${DATASET_DIR}/train.parquet}

if [[ ! -f "${TRAIN_FILES}" ]]; then
  echo "Train parquet not found: ${TRAIN_FILES}" >&2
  exit 1
fi

if [[ -z "${VAL_FILES:-}" ]]; then
  DEFAULT_VAL_FILES="${DATASET_DIR}/val.parquet"
  if [[ -f "${DEFAULT_VAL_FILES}" ]]; then
    VAL_FILES="${DEFAULT_VAL_FILES}"
  else
    VAL_FILES="${TRAIN_FILES}"
  fi
fi

export TRAIN_FILES
export VAL_FILES

# Keep outputs on /scratch and make the run easy to spot.
export RUN_ROOT=${RUN_ROOT:-${REPO_ROOT}/two_policy_runs}
export RUN_TAG=${RUN_TAG:-sample128_bs8_abs4_sol8_$(date -u +%Y%m%d_%H%M%S)}

# Use the raw solver requested by the user.
export SOLVER_MODEL_PATH=${SOLVER_MODEL_PATH:-Qwen/Qwen3-1.7B}

# Debug defaults for the requested rollout shape.
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-8}
export VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-8}
export NUM_ABS=${NUM_ABS:-4}
export NUM_SOL=${NUM_SOL:-8}
export VAL_NUM_ABS=${VAL_NUM_ABS:-1}
export VAL_NUM_SOL=${VAL_NUM_SOL:-1}
export ACTOR_PPO_MINI_BATCH_SIZE=${ACTOR_PPO_MINI_BATCH_SIZE:-32}

# Keep the GPU busy without paying the normal validation tax on every debug run.
export TRAINER_TOTAL_EPOCHS=${TRAINER_TOTAL_EPOCHS:-3}
export TRAINER_SAVE_FREQ=${TRAINER_SAVE_FREQ:-99999}
export TRAINER_TEST_FREQ=${TRAINER_TEST_FREQ:-0}
export TRAINER_VAL_BEFORE_TRAIN=${TRAINER_VAL_BEFORE_TRAIN:-False}
export TRAINER_PROJECT_NAME=${TRAINER_PROJECT_NAME:-verl_two_policy_debug}
export TRAINER_EXPERIMENT_NAME=${TRAINER_EXPERIMENT_NAME:-deepscaler_sample128_raw_qwen3_1p7b}
export TRAINER_RESUME_MODE=${TRAINER_RESUME_MODE:-disable}

# A100-safe defaults. These are tuned to keep the normal 8x4x8 rollout shape alive on a single A100.
export GPU_TUNING_PRESET=${GPU_TUNING_PRESET:-default}
export SOLVER_GPU_MEM_UTIL=${SOLVER_GPU_MEM_UTIL:-0.30}
export ABSTRACTION_GPU_MEM_UTIL=${ABSTRACTION_GPU_MEM_UTIL:-0.16}
export ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-32}
export SOLVER_MAX_BATCHED_TOKENS=${SOLVER_MAX_BATCHED_TOKENS:-4096}
export ABSTRACTION_MAX_BATCHED_TOKENS=${ABSTRACTION_MAX_BATCHED_TOKENS:-1024}
export ACTOR_PARAM_OFFLOAD=${ACTOR_PARAM_OFFLOAD:-True}
export ACTOR_OPTIMIZER_OFFLOAD=${ACTOR_OPTIMIZER_OFFLOAD:-True}
export ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU=${ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU:-16384}
export RAY_NUM_CPUS=${RAY_NUM_CPUS:-6}
export DATA_DATALOADER_NUM_WORKERS=${DATA_DATALOADER_NUM_WORKERS:-0}
export REWARD_NUM_WORKERS=${REWARD_NUM_WORKERS:-1}
export SOLVER_AGENT_NUM_WORKERS=${SOLVER_AGENT_NUM_WORKERS:-1}
export ABSTRACTION_AGENT_NUM_WORKERS=${ABSTRACTION_AGENT_NUM_WORKERS:-1}

export ABSTRACTION_MAX_RESP_LEN=${ABSTRACTION_MAX_RESP_LEN:-2048}
export SOLVER_MAX_RESP_LEN=${SOLVER_MAX_RESP_LEN:-8192}
export VALIDATION_SOLVER_MAX_RESP_LEN=${VALIDATION_SOLVER_MAX_RESP_LEN:-8192}
export USE_KL_LOSS=${USE_KL_LOSS:-True}

# Disable both layers of rollout filtering for this debug path.
export FILTER_GROUPS_ENABLE=${FILTER_GROUPS_ENABLE:-False}
export FILTER_GROUPS_METRIC=${FILTER_GROUPS_METRIC:-acc}
export FILTER_GROUPS_MAX_NUM_GEN_BATCHES=${FILTER_GROUPS_MAX_NUM_GEN_BATCHES:-1}
export TWO_POLICY_FILTER_SOLVER_GROUPS=${TWO_POLICY_FILTER_SOLVER_GROUPS:-False}

echo "Two-policy debug launch:"
echo "  dataset=${DATASET_DIR}"
echo "  train_files=${TRAIN_FILES}"
echo "  val_files=${VAL_FILES}"
echo "  solver_model=${SOLVER_MODEL_PATH}"
echo "  train_batch_size=${TRAIN_BATCH_SIZE}"
echo "  ppo_mini_batch_size=${ACTOR_PPO_MINI_BATCH_SIZE}"
echo "  num_abstractions=${NUM_ABS}"
echo "  num_solver_rollouts=${NUM_SOL}"
echo "  use_kl_loss=${USE_KL_LOSS}"
echo "  filter_groups=${FILTER_GROUPS_ENABLE}"
echo "  filter_solver_groups=${TWO_POLICY_FILTER_SOLVER_GROUPS}"
echo "  gpu_tuning_preset=${GPU_TUNING_PRESET}"
echo "  solver_max_resp_len=${SOLVER_MAX_RESP_LEN}"
echo "  abstraction_max_resp_len=${ABSTRACTION_MAX_RESP_LEN}"
echo "  rollout_max_num_seqs=${ROLLOUT_MAX_NUM_SEQS}"
echo "  solver_max_batched_tokens=${SOLVER_MAX_BATCHED_TOKENS}"
echo "  abstraction_max_batched_tokens=${ABSTRACTION_MAX_BATCHED_TOKENS}"
echo "  actor_param_offload=${ACTOR_PARAM_OFFLOAD}"
echo "  actor_optimizer_offload=${ACTOR_OPTIMIZER_OFFLOAD}"
echo "  data_dataloader_num_workers=${DATA_DATALOADER_NUM_WORKERS}"
echo "  reward_num_workers=${REWARD_NUM_WORKERS}"
echo "  solver_agent_num_workers=${SOLVER_AGENT_NUM_WORKERS}"
echo "  abstraction_agent_num_workers=${ABSTRACTION_AGENT_NUM_WORKERS}"

exec bash "${SCRIPT_DIR}/run_qwen_two_policy_full_dapo.sh" \
  two_policy.num_abstractions="${NUM_ABS}" \
  two_policy.num_solver_rollouts="${NUM_SOL}" \
  two_policy.validation_num_abstractions="${VAL_NUM_ABS}" \
  two_policy.validation_num_solver_rollouts="${VAL_NUM_SOL}" \
  two_policy.filter_solver_groups="${TWO_POLICY_FILTER_SOLVER_GROUPS}" \
  actor_rollout_ref.actor.use_kl_loss="${USE_KL_LOSS}" \
  abstraction_actor_rollout_ref.actor.use_kl_loss="${USE_KL_LOSS}" \
  data.dataloader_num_workers="${DATA_DATALOADER_NUM_WORKERS}" \
  reward.num_workers="${REWARD_NUM_WORKERS}" \
  actor_rollout_ref.rollout.agent.num_workers="${SOLVER_AGENT_NUM_WORKERS}" \
  abstraction_actor_rollout_ref.rollout.agent.num_workers="${ABSTRACTION_AGENT_NUM_WORKERS}" \
  algorithm.filter_groups.enable="${FILTER_GROUPS_ENABLE}" \
  algorithm.filter_groups.metric="${FILTER_GROUPS_METRIC}" \
  algorithm.filter_groups.max_num_gen_batches="${FILTER_GROUPS_MAX_NUM_GEN_BATCHES}" \
  trainer.resume_mode="${TRAINER_RESUME_MODE}" \
  "$@"
