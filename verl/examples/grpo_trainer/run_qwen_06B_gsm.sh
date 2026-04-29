#!/bin/bash
set -euo pipefail
set -x

export VLLM_ATTENTION_BACKEND=FLASH_ATTN


SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}

cd "${REPO_ROOT}"

GSM8K_TRAIN_PATH=${GSM8K_TRAIN_PATH:-${REPO_ROOT}/verl_data/gsm8k_baseline/train.parquet}
GSM8K_TEST_PATH=${GSM8K_TEST_PATH:-${REPO_ROOT}/verl_data/gsm8k_baseline/test.parquet}

if [[ ! -f "${GSM8K_TRAIN_PATH}" ]]; then
    echo "Train parquet not found: ${GSM8K_TRAIN_PATH}" >&2
    exit 1
fi

if [[ ! -f "${GSM8K_TEST_PATH}" ]]; then
    echo "Test parquet not found: ${GSM8K_TEST_PATH}" >&2
    exit 1
fi

TRAIN_FILES="['${GSM8K_TRAIN_PATH}']"
TEST_FILES="['${GSM8K_TEST_PATH}']"

TRAINER_PROJECT_NAME=${TRAINER_PROJECT_NAME:-verl_grpo_example_gsm8k_math}
TRAINER_EXPERIMENT_NAME=${TRAINER_EXPERIMENT_NAME:-qwen_0.6b_gsm_grpo}
RUN_ROOT=${RUN_ROOT:-${REPO_ROOT}/verl_checkpoints/gsm8k_qwen06b_grpo}
RUN_TAG=${RUN_TAG:-$(date -u +%Y%m%d_%H%M%S)}
RUN_DIR=${RUN_DIR:-${RUN_ROOT}/${TRAINER_EXPERIMENT_NAME}_${RUN_TAG}}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-${RUN_DIR}/checkpoints}
ROLLOUT_DIR=${ROLLOUT_DIR:-${RUN_DIR}/rollouts}
VALIDATION_DIR=${VALIDATION_DIR:-${RUN_DIR}/validation_rollouts}
WANDB_DIR=${WANDB_DIR:-${RUN_DIR}/wandb}

mkdir -p "${RUN_DIR}" "${CHECKPOINT_DIR}" "${ROLLOUT_DIR}" "${VALIDATION_DIR}" "${WANDB_DIR}"

export WANDB_DIR
export HYDRA_FULL_ERROR=1

echo "Qwen 0.6B GSM launch:"
echo "  train_files=${GSM8K_TRAIN_PATH}"
echo "  test_files=${GSM8K_TEST_PATH}"
echo "  run_root=${RUN_ROOT}"
echo "  run_tag=${RUN_TAG}"
echo "  run_dir=${RUN_DIR}"
echo "  checkpoint_dir=${CHECKPOINT_DIR}"
echo "  rollout_dir=${ROLLOUT_DIR}"
echo "  validation_dir=${VALIDATION_DIR}"
echo "  wandb_dir=${WANDB_DIR}"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${TEST_FILES}" \
    data.train_batch_size=8 \
    data.max_prompt_length=1024 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=Qwen/Qwen3-0.6B \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="${TRAINER_PROJECT_NAME}" \
    trainer.experiment_name="${TRAINER_EXPERIMENT_NAME}" \
    trainer.default_local_dir="${CHECKPOINT_DIR}" \
    trainer.rollout_data_dir="${ROLLOUT_DIR}" \
    trainer.validation_data_dir="${VALIDATION_DIR}" \
    ray_kwargs.ray_init.num_cpus=8 \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=1 \
    trainer.resume_mode=auto \
    trainer.max_actor_ckpt_to_keep=1 \
    trainer.max_critic_ckpt_to_keep=1 \
    trainer.nnodes=1 \
    reward.custom_reward_function.path="${REPO_ROOT}/verl/verl/utils/reward_score/deepscaler_math_reward_multibox_patched.py" \
    trainer.save_freq=100 \
    trainer.test_freq=100 \
    trainer.total_epochs=15 \
    "$@"
