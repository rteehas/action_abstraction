#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}

if [[ -z "${CONDA_DEFAULT_ENV:-}" ]] && [[ -f /workspace/miniconda3/etc/profile.d/conda.sh ]]; then
  source /workspace/miniconda3/etc/profile.d/conda.sh
  conda activate abstraction
fi

ray stop --force >/dev/null 2>&1 || true

MODE=${MODE:-grpo}
RUN_TAG=${RUN_TAG:-$(date -u +%Y%m%d_%H%M%S)}
DATA_DIR=${DATA_DIR:-${REPO_ROOT}/verl_data/two_policy_tiny_overfit}
TINY_OVERFIT_SOURCE_PARQUET=${TINY_OVERFIT_SOURCE_PARQUET:-${REPO_ROOT}/verl_data/sft_dataset_no_rl_partial_05_less1_traintest_concat/train.parquet}
MAX_PROMPT_LEN=${MAX_PROMPT_LEN:-4096}
ABSTRACTION_MAX_RESP_LEN=${ABSTRACTION_MAX_RESP_LEN:-1024}
SOLVER_MAX_RESP_LEN=${SOLVER_MAX_RESP_LEN:-4096}
MAX_RESP_LEN=$(( ABSTRACTION_MAX_RESP_LEN > SOLVER_MAX_RESP_LEN ? ABSTRACTION_MAX_RESP_LEN : SOLVER_MAX_RESP_LEN ))
TRAIN_INDICES=${TRAIN_INDICES:-144,1606}
VAL_INDICES=${VAL_INDICES:-144,1606}
ABSTRACTION_MODEL_PATH=${ABSTRACTION_MODEL_PATH:-${REPO_ROOT}/merged_models/qwen3_1_7b_principle_generator_ckpt1736}
SOLVER_MODEL_PATH=${SOLVER_MODEL_PATH:-Qwen/Qwen3-1.7B}
ABSTRACTION_PROMPT_TEMPLATE_PATH=${ABSTRACTION_PROMPT_TEMPLATE_PATH:-${REPO_ROOT}/prompt_templates/sft_principle_generation.txt}
SOLVER_PROMPT_TEMPLATE_PATH=${SOLVER_PROMPT_TEMPLATE_PATH:-${REPO_ROOT}/prompt_templates/hint_conditioned_problem_solving_rich_v1.txt}
GPU_TUNING_PRESET=${GPU_TUNING_PRESET:-default}
RUN_ROOT=${RUN_ROOT:-/tmp/action_abstraction/two_policy_runs}
RUN_DIR=${RUN_DIR:-${RUN_ROOT}/tiny_overfit_${MODE}_${RUN_TAG}}
ROLLOUT_DIR=${ROLLOUT_DIR:-${RUN_DIR}/rollouts}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-${RUN_DIR}/checkpoints}
TMP_ROOT=${TMP_ROOT:-/scratch/rst306/ray_tmp}
RAY_NUM_CPUS=${RAY_NUM_CPUS:-8}
DATA_DATALOADER_NUM_WORKERS=${DATA_DATALOADER_NUM_WORKERS:-0}
REWARD_NUM_WORKERS=${REWARD_NUM_WORKERS:-1}
EPOCHS=${EPOCHS:-20}
LR=${LR:-1e-6}
NUM_ABS=${NUM_ABS:-2}
NUM_SOL=${NUM_SOL:-4}
RUN_BASELINE_EVAL=${RUN_BASELINE_EVAL:-True}
RUN_TRAIN=${RUN_TRAIN:-True}
RUN_FINAL_EVAL=${RUN_FINAL_EVAL:-True}
TRAIN_VAL_BEFORE_TRAIN=${TRAIN_VAL_BEFORE_TRAIN:-True}
TWO_POLICY_DECOUPLED_SOLVER_SCHEDULE_ENABLE=${TWO_POLICY_DECOUPLED_SOLVER_SCHEDULE_ENABLE:-False}
TWO_POLICY_SOLVER_UPDATE_EVERY_N_STEPS=${TWO_POLICY_SOLVER_UPDATE_EVERY_N_STEPS:-1}
TWO_POLICY_NON_UPDATE_SOLVER_ROLLOUTS=${TWO_POLICY_NON_UPDATE_SOLVER_ROLLOUTS:-1}
TWO_POLICY_NON_UPDATE_SOLVER_TEMPERATURE=${TWO_POLICY_NON_UPDATE_SOLVER_TEMPERATURE:-0.0}
TWO_POLICY_NON_UPDATE_SOLVER_TOP_P=${TWO_POLICY_NON_UPDATE_SOLVER_TOP_P:-1.0}
TWO_POLICY_NON_UPDATE_SOLVER_TOP_K=${TWO_POLICY_NON_UPDATE_SOLVER_TOP_K:--1}
TWO_POLICY_CONTROL_ABSTRACTION_ENABLE=${TWO_POLICY_CONTROL_ABSTRACTION_ENABLE:-False}
TWO_POLICY_CONTROL_ABSTRACTION_TEXT=${TWO_POLICY_CONTROL_ABSTRACTION_TEXT:-Identify the givens, translate the problem into mathematical relations, and solve step by step.}
TWO_POLICY_CONTROL_ABSTRACTION_INCLUDE_IN_VALIDATION=${TWO_POLICY_CONTROL_ABSTRACTION_INCLUDE_IN_VALIDATION:-False}
TWO_POLICY_VM_REWARD_ENABLE=${TWO_POLICY_VM_REWARD_ENABLE:-False}
TWO_POLICY_VM_REWARD_TRANSFORM=${TWO_POLICY_VM_REWARD_TRANSFORM:-logit}
TWO_POLICY_VM_REWARD_SOLVER_WEIGHT=${TWO_POLICY_VM_REWARD_SOLVER_WEIGHT:-0.1}
TWO_POLICY_VM_REWARD_ABSTRACTION_WEIGHT=${TWO_POLICY_VM_REWARD_ABSTRACTION_WEIGHT:-0.1}

case "${GPU_TUNING_PRESET}" in
  default)
    DEFAULT_SOLVER_GPU_MEM_UTIL=0.45
    DEFAULT_ABSTRACTION_GPU_MEM_UTIL=0.45
    DEFAULT_ROLLOUT_MAX_NUM_SEQS=1024
    DEFAULT_SOLVER_MAX_BATCHED_TOKENS=8192
    DEFAULT_ABSTRACTION_MAX_BATCHED_TOKENS=8192
    ;;
  h200_balanced)
    DEFAULT_SOLVER_GPU_MEM_UTIL=0.50
    DEFAULT_ABSTRACTION_GPU_MEM_UTIL=0.28
    DEFAULT_ROLLOUT_MAX_NUM_SEQS=144
    DEFAULT_SOLVER_MAX_BATCHED_TOKENS=10240
    DEFAULT_ABSTRACTION_MAX_BATCHED_TOKENS=2560
    ;;
  h200_aggressive)
    DEFAULT_SOLVER_GPU_MEM_UTIL=0.55
    DEFAULT_ABSTRACTION_GPU_MEM_UTIL=0.30
    DEFAULT_ROLLOUT_MAX_NUM_SEQS=160
    DEFAULT_SOLVER_MAX_BATCHED_TOKENS=12288
    DEFAULT_ABSTRACTION_MAX_BATCHED_TOKENS=3072
    ;;
  *)
    echo "Unknown GPU_TUNING_PRESET: ${GPU_TUNING_PRESET}" >&2
    exit 1
    ;;
esac

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-1}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-2}
ACTOR_PPO_MINI_BATCH_SIZE=${ACTOR_PPO_MINI_BATCH_SIZE:-1}
ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU=${ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU:-32768}
SOLVER_GPU_MEM_UTIL=${SOLVER_GPU_MEM_UTIL:-${DEFAULT_SOLVER_GPU_MEM_UTIL}}
ABSTRACTION_GPU_MEM_UTIL=${ABSTRACTION_GPU_MEM_UTIL:-${DEFAULT_ABSTRACTION_GPU_MEM_UTIL}}
ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-${DEFAULT_ROLLOUT_MAX_NUM_SEQS}}
SOLVER_MAX_BATCHED_TOKENS=${SOLVER_MAX_BATCHED_TOKENS:-${DEFAULT_SOLVER_MAX_BATCHED_TOKENS}}
ABSTRACTION_MAX_BATCHED_TOKENS=${ABSTRACTION_MAX_BATCHED_TOKENS:-${DEFAULT_ABSTRACTION_MAX_BATCHED_TOKENS}}
ACTOR_PARAM_OFFLOAD=${ACTOR_PARAM_OFFLOAD:-False}
ACTOR_OPTIMIZER_OFFLOAD=${ACTOR_OPTIMIZER_OFFLOAD:-False}
ROLLOUT_ENFORCE_EAGER=${ROLLOUT_ENFORCE_EAGER:-True}
ROLLOUT_FREE_CACHE_ENGINE=${ROLLOUT_FREE_CACHE_ENGINE:-True}
SOLVER_UPDATE_WEIGHTS_BUCKET_MB=${SOLVER_UPDATE_WEIGHTS_BUCKET_MB:-2048}
ABSTRACTION_UPDATE_WEIGHTS_BUCKET_MB=${ABSTRACTION_UPDATE_WEIGHTS_BUCKET_MB:-2048}
TWO_POLICY_FILTER_SOLVER_GROUPS=${TWO_POLICY_FILTER_SOLVER_GROUPS:-True}
MODEL_USE_LIGER=${MODEL_USE_LIGER:-False}
TRAIN_INDICES_REPEAT=${TRAIN_INDICES_REPEAT:-1}
VAL_INDICES_REPEAT=${VAL_INDICES_REPEAT:-1}

repeat_csv_indices() {
  local csv="$1"
  local repeat="$2"
  local -a parts expanded=()
  local part
  local i

  if [[ -z "${repeat}" ]] || (( repeat < 1 )); then
    repeat=1
  fi

  IFS=',' read -r -a parts <<< "${csv}"
  for ((i = 0; i < repeat; i++)); do
    for part in "${parts[@]}"; do
      part="${part//[[:space:]]/}"
      if [[ -n "${part}" ]]; then
        expanded+=("${part}")
      fi
    done
  done

  local IFS=,
  printf '%s\n' "${expanded[*]}"
}

count_csv_indices() {
  local csv="$1"
  local -a parts
  local part
  local count=0
  IFS=',' read -r -a parts <<< "${csv}"
  for part in "${parts[@]}"; do
    part="${part//[[:space:]]/}"
    if [[ -n "${part}" ]]; then
      ((count += 1))
    fi
  done
  printf '%s\n' "${count}"
}

EFFECTIVE_TRAIN_INDICES=$(repeat_csv_indices "${TRAIN_INDICES}" "${TRAIN_INDICES_REPEAT}")
EFFECTIVE_VAL_INDICES=$(repeat_csv_indices "${VAL_INDICES}" "${VAL_INDICES_REPEAT}")
TRAIN_ROW_COUNT=$(count_csv_indices "${EFFECTIVE_TRAIN_INDICES}")

if (( TRAIN_BATCH_SIZE > TRAIN_ROW_COUNT )); then
  echo "TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE} exceeds tiny train set size ${TRAIN_ROW_COUNT}." >&2
  echo "Increase TRAIN_INDICES_REPEAT or lower TRAIN_BATCH_SIZE." >&2
  exit 2
fi

# Ray falls back to /tmp by default, but /tmp may be missing inside the container.
export TMPDIR=${TMPDIR:-${TMP_ROOT}}
export RAY_TMPDIR=${RAY_TMPDIR:-${TMPDIR}}

mkdir -p "${RUN_DIR}" "${ROLLOUT_DIR}" "${CHECKPOINT_DIR}" "${TMPDIR}" "${RAY_TMPDIR}"

if [ ! -d "${ABSTRACTION_MODEL_PATH}" ] || [ ! -f "${ABSTRACTION_MODEL_PATH}/config.json" ]; then
  python "${REPO_ROOT}/scripts/merge_lora_adapter.py" \
    --output-dir "${ABSTRACTION_MODEL_PATH}"
fi

python "${REPO_ROOT}/scripts/build_two_policy_tiny_rl_dataset.py" \
  --source-parquet "${TINY_OVERFIT_SOURCE_PARQUET}" \
  --output-dir "${DATA_DIR}" \
  --train-indices "${EFFECTIVE_TRAIN_INDICES}" \
  --val-indices "${EFFECTIVE_VAL_INDICES}"

echo "Tiny overfit throughput config:"
echo "  GPU_TUNING_PRESET=${GPU_TUNING_PRESET}"
echo "  TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE}"
echo "  ACTOR_PPO_MINI_BATCH_SIZE=${ACTOR_PPO_MINI_BATCH_SIZE}"
echo "  TRAIN_INDICES_REPEAT=${TRAIN_INDICES_REPEAT}"
echo "  TRAIN_ROW_COUNT=${TRAIN_ROW_COUNT}"
echo "  SOLVER_GPU_MEM_UTIL=${SOLVER_GPU_MEM_UTIL}"
echo "  ABSTRACTION_GPU_MEM_UTIL=${ABSTRACTION_GPU_MEM_UTIL}"
echo "  SOLVER_MAX_BATCHED_TOKENS=${SOLVER_MAX_BATCHED_TOKENS}"
echo "  ABSTRACTION_MAX_BATCHED_TOKENS=${ABSTRACTION_MAX_BATCHED_TOKENS}"

cd "${REPO_ROOT}/verl"

MAIN_MODULE=recipe.two_policy.main_two_policy_ppo
reward_manager=naive
extra_args=()
if [ "${MODE}" = "dapo" ]; then
  MAIN_MODULE=recipe.two_policy.main_two_policy_dapo
  reward_manager=dapo
  extra_args+=(algorithm.filter_groups.enable=True)
  extra_args+=(algorithm.filter_groups.metric=acc)
  extra_args+=(algorithm.filter_groups.max_num_gen_batches=4)
  extra_args+=(reward.reward_kwargs.max_resp_len="${SOLVER_MAX_RESP_LEN}")
fi

common_args=(
  hydra.searchpath=[file://${REPO_ROOT}/verl/verl/trainer/config]
  algorithm.adv_estimator=grpo
  data.train_files="${DATA_DIR}/train.parquet"
  data.val_files="${DATA_DIR}/val.parquet"
  data.train_batch_size="${TRAIN_BATCH_SIZE}"
  data.val_batch_size="${VAL_BATCH_SIZE}"
  data.shuffle=False
  data.dataloader_num_workers="${DATA_DATALOADER_NUM_WORKERS}"
  data.max_prompt_length="${MAX_PROMPT_LEN}"
  data.max_response_length="${MAX_RESP_LEN}"
  actor_rollout_ref.model.path="${SOLVER_MODEL_PATH}"
  actor_rollout_ref.model.lora_rank=32
  actor_rollout_ref.model.lora_alpha=32
  actor_rollout_ref.model.target_modules=all-linear
  actor_rollout_ref.model.use_remove_padding=True
  actor_rollout_ref.model.enable_gradient_checkpointing=True
  actor_rollout_ref.model.use_liger="${MODEL_USE_LIGER}"
  actor_rollout_ref.actor.optim.lr="${LR}"
  actor_rollout_ref.actor.entropy_coeff=0
  actor_rollout_ref.actor.ppo_mini_batch_size="${ACTOR_PPO_MINI_BATCH_SIZE}"
  actor_rollout_ref.actor.use_dynamic_bsz=True
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU}"
  actor_rollout_ref.actor.use_kl_loss=True
  actor_rollout_ref.actor.kl_loss_coef=0.001
  actor_rollout_ref.actor.kl_loss_type=low_var_kl
  actor_rollout_ref.actor.ulysses_sequence_parallel_size=1
  actor_rollout_ref.actor.fsdp_config.param_offload="${ACTOR_PARAM_OFFLOAD}"
  +actor_rollout_ref.actor.fsdp_config.grad_offload=False
  actor_rollout_ref.actor.fsdp_config.optimizer_offload="${ACTOR_OPTIMIZER_OFFLOAD}"
  actor_rollout_ref.rollout.tensor_model_parallel_size=1
  actor_rollout_ref.rollout.name=vllm
  actor_rollout_ref.rollout.dtype=bfloat16
  actor_rollout_ref.rollout.temperature=0.6
  actor_rollout_ref.rollout.gpu_memory_utilization="${SOLVER_GPU_MEM_UTIL}"
  actor_rollout_ref.rollout.n=1
  actor_rollout_ref.rollout.max_num_seqs="${ROLLOUT_MAX_NUM_SEQS}"
  actor_rollout_ref.rollout.max_num_batched_tokens="${SOLVER_MAX_BATCHED_TOKENS}"
  actor_rollout_ref.rollout.response_length="${SOLVER_MAX_RESP_LEN}"
  actor_rollout_ref.rollout.enforce_eager="${ROLLOUT_ENFORCE_EAGER}"
  actor_rollout_ref.rollout.free_cache_engine="${ROLLOUT_FREE_CACHE_ENGINE}"
  actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes="${SOLVER_UPDATE_WEIGHTS_BUCKET_MB}"
  actor_rollout_ref.rollout.load_format=safetensors
  actor_rollout_ref.ref.fsdp_config.param_offload=True
  abstraction_actor_rollout_ref.model.path="${ABSTRACTION_MODEL_PATH}"
  abstraction_actor_rollout_ref.model.lora_rank=32
  abstraction_actor_rollout_ref.model.lora_alpha=32
  abstraction_actor_rollout_ref.model.target_modules=all-linear
  abstraction_actor_rollout_ref.model.use_remove_padding=True
  abstraction_actor_rollout_ref.model.enable_gradient_checkpointing=True
  abstraction_actor_rollout_ref.model.use_liger="${MODEL_USE_LIGER}"
  abstraction_actor_rollout_ref.actor.optim.lr="${LR}"
  abstraction_actor_rollout_ref.actor.entropy_coeff=0
  abstraction_actor_rollout_ref.actor.ppo_mini_batch_size="${ACTOR_PPO_MINI_BATCH_SIZE}"
  abstraction_actor_rollout_ref.actor.use_dynamic_bsz=True
  abstraction_actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU}"
  abstraction_actor_rollout_ref.actor.use_kl_loss=True
  abstraction_actor_rollout_ref.actor.kl_loss_coef=0.001
  abstraction_actor_rollout_ref.actor.kl_loss_type=low_var_kl
  abstraction_actor_rollout_ref.actor.ulysses_sequence_parallel_size=1
  abstraction_actor_rollout_ref.actor.fsdp_config.param_offload="${ACTOR_PARAM_OFFLOAD}"
  +abstraction_actor_rollout_ref.actor.fsdp_config.grad_offload=False
  abstraction_actor_rollout_ref.actor.fsdp_config.optimizer_offload="${ACTOR_OPTIMIZER_OFFLOAD}"
  abstraction_actor_rollout_ref.rollout.tensor_model_parallel_size=1
  abstraction_actor_rollout_ref.rollout.name=vllm
  abstraction_actor_rollout_ref.rollout.dtype=bfloat16
  abstraction_actor_rollout_ref.rollout.temperature=0.6
  abstraction_actor_rollout_ref.rollout.gpu_memory_utilization="${ABSTRACTION_GPU_MEM_UTIL}"
  abstraction_actor_rollout_ref.rollout.n=1
  abstraction_actor_rollout_ref.rollout.max_num_seqs="${ROLLOUT_MAX_NUM_SEQS}"
  abstraction_actor_rollout_ref.rollout.max_num_batched_tokens="${ABSTRACTION_MAX_BATCHED_TOKENS}"
  abstraction_actor_rollout_ref.rollout.response_length="${ABSTRACTION_MAX_RESP_LEN}"
  abstraction_actor_rollout_ref.rollout.enforce_eager="${ROLLOUT_ENFORCE_EAGER}"
  abstraction_actor_rollout_ref.rollout.free_cache_engine="${ROLLOUT_FREE_CACHE_ENGINE}"
  abstraction_actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes="${ABSTRACTION_UPDATE_WEIGHTS_BUCKET_MB}"
  abstraction_actor_rollout_ref.rollout.load_format=safetensors
  abstraction_actor_rollout_ref.ref.fsdp_config.param_offload=True
  two_policy.abstraction_prompt_template_path="${ABSTRACTION_PROMPT_TEMPLATE_PATH}"
  two_policy.solver_prompt_template_path="${SOLVER_PROMPT_TEMPLATE_PATH}"
  two_policy.num_abstractions="${NUM_ABS}"
  two_policy.num_solver_rollouts="${NUM_SOL}"
  two_policy.validation_num_abstractions="${NUM_ABS}"
  two_policy.validation_num_solver_rollouts="${NUM_SOL}"
  two_policy.filter_solver_groups="${TWO_POLICY_FILTER_SOLVER_GROUPS}"
  two_policy.decoupled_solver_schedule.enable="${TWO_POLICY_DECOUPLED_SOLVER_SCHEDULE_ENABLE}"
  two_policy.decoupled_solver_schedule.solver_update_every_n_steps="${TWO_POLICY_SOLVER_UPDATE_EVERY_N_STEPS}"
  two_policy.decoupled_solver_schedule.non_update_solver_rollouts="${TWO_POLICY_NON_UPDATE_SOLVER_ROLLOUTS}"
  two_policy.decoupled_solver_schedule.non_update_solver_temperature="${TWO_POLICY_NON_UPDATE_SOLVER_TEMPERATURE}"
  two_policy.decoupled_solver_schedule.non_update_solver_top_p="${TWO_POLICY_NON_UPDATE_SOLVER_TOP_P}"
  two_policy.decoupled_solver_schedule.non_update_solver_top_k="${TWO_POLICY_NON_UPDATE_SOLVER_TOP_K}"
  two_policy.control_abstraction.enable="${TWO_POLICY_CONTROL_ABSTRACTION_ENABLE}"
  two_policy.control_abstraction.text="${TWO_POLICY_CONTROL_ABSTRACTION_TEXT}"
  two_policy.control_abstraction.include_in_validation="${TWO_POLICY_CONTROL_ABSTRACTION_INCLUDE_IN_VALIDATION}"
  two_policy.vm_reward.enable="${TWO_POLICY_VM_REWARD_ENABLE}"
  two_policy.vm_reward.transform="${TWO_POLICY_VM_REWARD_TRANSFORM}"
  two_policy.vm_reward.solver_weight="${TWO_POLICY_VM_REWARD_SOLVER_WEIGHT}"
  two_policy.vm_reward.abstraction_weight="${TWO_POLICY_VM_REWARD_ABSTRACTION_WEIGHT}"
  trainer.critic_warmup=0
  trainer.logger=[console,wandb]
  trainer.project_name=verl_two_policy_overfit
  trainer.experiment_name="tiny_${MODE}_${RUN_TAG}"
  trainer.n_gpus_per_node=1
  trainer.nnodes=1
  trainer.rollout_data_dir="${ROLLOUT_DIR}"
  trainer.default_local_dir="${CHECKPOINT_DIR}"
  trainer.save_freq=99999
  trainer.test_freq=99999
  trainer.max_actor_ckpt_to_keep=4
  reward.custom_reward_function.path="${REPO_ROOT}/verl/verl/utils/reward_score/deepscaler_math_reward_multibox_patched.py"
  reward.custom_reward_function.name=compute_score
  reward.num_workers="${REWARD_NUM_WORKERS}"
  reward.reward_manager.name="${reward_manager}"
  algorithm.use_kl_in_reward=False
  ray_kwargs.ray_init.num_cpus="${RAY_NUM_CPUS}"
)

if [ "${RUN_BASELINE_EVAL}" = "True" ]; then
  python -m "${MAIN_MODULE}" \
    "${common_args[@]}" \
    trainer.val_before_train=True \
    trainer.val_only=True \
    trainer.resume_mode=disable \
    "${extra_args[@]}" \
    "$@" | tee "${RUN_DIR}/baseline_eval.log"

  ray stop --force >/dev/null 2>&1 || true
fi

if [ "${RUN_TRAIN}" = "True" ]; then
  ray stop --force >/dev/null 2>&1 || true
  python -m "${MAIN_MODULE}" \
    "${common_args[@]}" \
    trainer.val_before_train="${TRAIN_VAL_BEFORE_TRAIN}" \
    trainer.val_only=False \
    trainer.total_epochs="${EPOCHS}" \
    trainer.resume_mode=disable \
    "${extra_args[@]}" \
    "$@" | tee "${RUN_DIR}/train.log"

  ray stop --force >/dev/null 2>&1 || true
fi

if [ "${RUN_FINAL_EVAL}" = "True" ]; then
  LATEST_CKPT=$(find "${CHECKPOINT_DIR}" -maxdepth 1 -type d -name "global_step_*" | sort -V | tail -n 1)
  if [ -z "${LATEST_CKPT}" ]; then
    echo "No checkpoint found under ${CHECKPOINT_DIR}" >&2
    exit 1
  fi

  python -m "${MAIN_MODULE}" \
    "${common_args[@]}" \
    trainer.val_before_train=True \
    trainer.val_only=True \
    trainer.resume_mode=resume_path \
    trainer.resume_from_path="${LATEST_CKPT}" \
    "${extra_args[@]}" \
    "$@" | tee "${RUN_DIR}/final_eval.log"
fi

if [ "${RUN_BASELINE_EVAL}" = "True" ] && [ "${RUN_FINAL_EVAL}" = "True" ]; then
  python - <<"PY" "${RUN_DIR}/baseline_eval.log" "${RUN_DIR}/final_eval.log"
import ast
import re
import sys
from pathlib import Path


def parse_metrics(path: str):
    text = Path(path).read_text()
    matches = re.findall(r"Initial validation metrics: (\{.*?\})", text)
    if not matches:
        raise SystemExit(f"Could not find validation metrics in {path}")
    return ast.literal_eval(matches[-1])

baseline = parse_metrics(sys.argv[1])
final = parse_metrics(sys.argv[2])
keys = [
    "val/problem_best_reward_mean",
    "val/problem_mean_reward_mean",
    "val/problem_best_acc_mean",
    "val/problem_mean_acc_mean",
    "val/abstraction_valid_rate",
]
print("Baseline metrics:")
for key in keys:
    if key in baseline:
        print(f"  {key}: {baseline[key]}")
print("Final metrics:")
for key in keys:
    if key in final:
        print(f"  {key}: {final[key]}")
print("Delta:")
for key in keys:
    if key in baseline and key in final:
        print(f"  {key}: {final[key] - baseline[key]}")
PY
fi
