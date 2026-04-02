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
SOLVER_MODEL_PATH=${SOLVER_MODEL_PATH:-${REPO_ROOT}/solver_sft_qwen_1_7b_450}
ABSTRACTION_PROMPT_TEMPLATE_PATH=${ABSTRACTION_PROMPT_TEMPLATE_PATH:-${REPO_ROOT}/prompt_templates/sft_principle_generation.txt}
SOLVER_PROMPT_TEMPLATE_PATH=${SOLVER_PROMPT_TEMPLATE_PATH:-${REPO_ROOT}/prompt_templates/hint_conditioned_problem_solving_rich_v1.txt}
RUN_ROOT=${RUN_ROOT:-/tmp/action_abstraction/two_policy_runs}
RUN_DIR=${RUN_DIR:-${RUN_ROOT}/tiny_overfit_${MODE}_${RUN_TAG}}
ROLLOUT_DIR=${ROLLOUT_DIR:-${RUN_DIR}/rollouts}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-${RUN_DIR}/checkpoints}
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

mkdir -p "${RUN_DIR}" "${ROLLOUT_DIR}" "${CHECKPOINT_DIR}"

if [ ! -d "${ABSTRACTION_MODEL_PATH}" ] || [ ! -f "${ABSTRACTION_MODEL_PATH}/config.json" ]; then
  python "${REPO_ROOT}/scripts/merge_lora_adapter.py" \
    --output-dir "${ABSTRACTION_MODEL_PATH}"
fi

python "${REPO_ROOT}/scripts/build_two_policy_tiny_rl_dataset.py" \
  --source-parquet "${TINY_OVERFIT_SOURCE_PARQUET}" \
  --output-dir "${DATA_DIR}" \
  --train-indices "${TRAIN_INDICES}" \
  --val-indices "${VAL_INDICES}"

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
  data.train_batch_size=1
  data.val_batch_size=2
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
  actor_rollout_ref.actor.optim.lr="${LR}"
  actor_rollout_ref.actor.entropy_coeff=0
  actor_rollout_ref.actor.ppo_mini_batch_size=1
  actor_rollout_ref.actor.use_dynamic_bsz=True
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768
  actor_rollout_ref.actor.use_kl_loss=True
  actor_rollout_ref.actor.kl_loss_coef=0.001
  actor_rollout_ref.actor.kl_loss_type=low_var_kl
  actor_rollout_ref.actor.ulysses_sequence_parallel_size=1
  actor_rollout_ref.actor.fsdp_config.param_offload=False
  +actor_rollout_ref.actor.fsdp_config.grad_offload=False
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
  actor_rollout_ref.rollout.tensor_model_parallel_size=1
  actor_rollout_ref.rollout.name=vllm
  actor_rollout_ref.rollout.dtype=bfloat16
  actor_rollout_ref.rollout.temperature=0.6
  actor_rollout_ref.rollout.gpu_memory_utilization=0.45
  actor_rollout_ref.rollout.n=1
  actor_rollout_ref.rollout.response_length="${SOLVER_MAX_RESP_LEN}"
  actor_rollout_ref.rollout.enforce_eager=True
  actor_rollout_ref.rollout.free_cache_engine=True
  actor_rollout_ref.rollout.load_format=safetensors
  actor_rollout_ref.ref.fsdp_config.param_offload=True
  abstraction_actor_rollout_ref.model.path="${ABSTRACTION_MODEL_PATH}"
  abstraction_actor_rollout_ref.model.lora_rank=32
  abstraction_actor_rollout_ref.model.lora_alpha=32
  abstraction_actor_rollout_ref.model.target_modules=all-linear
  abstraction_actor_rollout_ref.model.use_remove_padding=True
  abstraction_actor_rollout_ref.model.enable_gradient_checkpointing=True
  abstraction_actor_rollout_ref.actor.optim.lr="${LR}"
  abstraction_actor_rollout_ref.actor.entropy_coeff=0
  abstraction_actor_rollout_ref.actor.ppo_mini_batch_size=1
  abstraction_actor_rollout_ref.actor.use_dynamic_bsz=True
  abstraction_actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768
  abstraction_actor_rollout_ref.actor.use_kl_loss=True
  abstraction_actor_rollout_ref.actor.kl_loss_coef=0.001
  abstraction_actor_rollout_ref.actor.kl_loss_type=low_var_kl
  abstraction_actor_rollout_ref.actor.ulysses_sequence_parallel_size=1
  abstraction_actor_rollout_ref.actor.fsdp_config.param_offload=False
  +abstraction_actor_rollout_ref.actor.fsdp_config.grad_offload=False
  abstraction_actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
  abstraction_actor_rollout_ref.rollout.tensor_model_parallel_size=1
  abstraction_actor_rollout_ref.rollout.name=vllm
  abstraction_actor_rollout_ref.rollout.dtype=bfloat16
  abstraction_actor_rollout_ref.rollout.temperature=0.6
  abstraction_actor_rollout_ref.rollout.gpu_memory_utilization=0.45
  abstraction_actor_rollout_ref.rollout.n=1
  abstraction_actor_rollout_ref.rollout.response_length="${ABSTRACTION_MAX_RESP_LEN}"
  abstraction_actor_rollout_ref.rollout.enforce_eager=True
  abstraction_actor_rollout_ref.rollout.free_cache_engine=True
  abstraction_actor_rollout_ref.rollout.load_format=safetensors
  abstraction_actor_rollout_ref.ref.fsdp_config.param_offload=True
  two_policy.abstraction_prompt_template_path="${ABSTRACTION_PROMPT_TEMPLATE_PATH}"
  two_policy.solver_prompt_template_path="${SOLVER_PROMPT_TEMPLATE_PATH}"
  two_policy.num_abstractions="${NUM_ABS}"
  two_policy.num_solver_rollouts="${NUM_SOL}"
  two_policy.validation_num_abstractions="${NUM_ABS}"
  two_policy.validation_num_solver_rollouts="${NUM_SOL}"
  two_policy.decoupled_solver_schedule.enable="${TWO_POLICY_DECOUPLED_SOLVER_SCHEDULE_ENABLE}"
  two_policy.decoupled_solver_schedule.solver_update_every_n_steps="${TWO_POLICY_SOLVER_UPDATE_EVERY_N_STEPS}"
  two_policy.decoupled_solver_schedule.non_update_solver_rollouts="${TWO_POLICY_NON_UPDATE_SOLVER_ROLLOUTS}"
  two_policy.decoupled_solver_schedule.non_update_solver_temperature="${TWO_POLICY_NON_UPDATE_SOLVER_TEMPERATURE}"
  two_policy.decoupled_solver_schedule.non_update_solver_top_p="${TWO_POLICY_NON_UPDATE_SOLVER_TOP_P}"
  two_policy.decoupled_solver_schedule.non_update_solver_top_k="${TWO_POLICY_NON_UPDATE_SOLVER_TOP_K}"
  trainer.critic_warmup=0
  trainer.logger=[console]
  trainer.project_name=verl_two_policy_overfit
  trainer.experiment_name="tiny_${MODE}_${RUN_TAG}"
  trainer.n_gpus_per_node=1
  trainer.nnodes=1
  trainer.rollout_data_dir="${ROLLOUT_DIR}"
  trainer.default_local_dir="${CHECKPOINT_DIR}"
  trainer.save_freq=1
  trainer.test_freq=1
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
