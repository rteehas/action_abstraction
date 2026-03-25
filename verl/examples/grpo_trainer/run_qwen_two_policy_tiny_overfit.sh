#!/bin/bash
set -euo pipefail

source /workspace/miniconda3/etc/profile.d/conda.sh
conda activate abstraction

ray stop --force >/dev/null 2>&1 || true

MODE=${MODE:-grpo}
RUN_TAG=${RUN_TAG:-$(date -u +%Y%m%d_%H%M%S)}
DATA_DIR=${DATA_DIR:-/workspace/action_abstraction/verl_data/two_policy_tiny_overfit}
MAX_PROMPT_LEN=${MAX_PROMPT_LEN:-4096}
ABSTRACTION_MAX_RESP_LEN=${ABSTRACTION_MAX_RESP_LEN:-1024}
SOLVER_MAX_RESP_LEN=${SOLVER_MAX_RESP_LEN:-4096}
MAX_RESP_LEN=$(( ABSTRACTION_MAX_RESP_LEN > SOLVER_MAX_RESP_LEN ? ABSTRACTION_MAX_RESP_LEN : SOLVER_MAX_RESP_LEN ))
TRAIN_INDICES=${TRAIN_INDICES:-1606,2758}
VAL_INDICES=${VAL_INDICES:-1606,2758}
ABSTRACTION_MODEL_PATH=${ABSTRACTION_MODEL_PATH:-/workspace/action_abstraction/merged_models/qwen3_1_7b_principle_generator_ckpt1736}
SOLVER_MODEL_PATH=${SOLVER_MODEL_PATH:-Qwen/Qwen3-1.7B}
RUN_ROOT=${RUN_ROOT:-/tmp/action_abstraction/two_policy_runs}
RUN_DIR=${RUN_DIR:-${RUN_ROOT}/tiny_overfit_${MODE}_${RUN_TAG}}
ROLLOUT_DIR=${ROLLOUT_DIR:-${RUN_DIR}/rollouts}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-${RUN_DIR}/checkpoints}
EPOCHS=${EPOCHS:-20}
LR=${LR:-1e-6}
NUM_ABS=${NUM_ABS:-2}
NUM_SOL=${NUM_SOL:-4}

mkdir -p "${RUN_DIR}" "${ROLLOUT_DIR}" "${CHECKPOINT_DIR}"

if [ ! -d "${ABSTRACTION_MODEL_PATH}" ] || [ ! -f "${ABSTRACTION_MODEL_PATH}/config.json" ]; then
  python /workspace/action_abstraction/scripts/merge_lora_adapter.py \
    --output-dir "${ABSTRACTION_MODEL_PATH}"
fi

python /workspace/action_abstraction/scripts/build_two_policy_tiny_rl_dataset.py \
  --output-dir "${DATA_DIR}" \
  --train-indices "${TRAIN_INDICES}" \
  --val-indices "${VAL_INDICES}"

cd /workspace/action_abstraction/verl

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
  hydra.searchpath=[file:///workspace/action_abstraction/verl/verl/trainer/config]
  algorithm.adv_estimator=grpo
  data.train_files="${DATA_DIR}/train.parquet"
  data.val_files="${DATA_DIR}/val.parquet"
  data.train_batch_size=1
  data.val_batch_size=2
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
  two_policy.num_abstractions="${NUM_ABS}"
  two_policy.num_solver_rollouts="${NUM_SOL}"
  two_policy.validation_num_abstractions="${NUM_ABS}"
  two_policy.validation_num_solver_rollouts="${NUM_SOL}"
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
  reward.custom_reward_function.path=/workspace/action_abstraction/verl/verl/utils/reward_score/deepscaler_math_reward_multibox_patched.py
  reward.custom_reward_function.name=compute_score
  reward.reward_manager.name="${reward_manager}"
  algorithm.use_kl_in_reward=False
  ray_kwargs.ray_init.num_cpus=32
)

python -m "${MAIN_MODULE}" \
  "${common_args[@]}" \
  trainer.val_before_train=True \
  trainer.val_only=True \
  trainer.resume_mode=disable \
  "${extra_args[@]}" \
  "$@" | tee "${RUN_DIR}/baseline_eval.log"

ray stop --force >/dev/null 2>&1 || true

ray stop --force >/dev/null 2>&1 || true
python -m "${MAIN_MODULE}" \
  "${common_args[@]}" \
  trainer.val_before_train=True \
  trainer.val_only=False \
  trainer.total_epochs="${EPOCHS}" \
  trainer.resume_mode=disable \
  "${extra_args[@]}" \
  "$@" | tee "${RUN_DIR}/train.log"

ray stop --force >/dev/null 2>&1 || true

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
