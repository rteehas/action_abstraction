#!/bin/bash
set -euo pipefail
set -x

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}

cd "${REPO_ROOT}/verl"

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1

ray stop --force >/dev/null 2>&1 || true

ABSTRACTION_MODEL_PATH=${REPO_ROOT}/merged_models/qwen3_1_7b_principle_generator_ckpt1736
TRAIN_FILES=${REPO_ROOT}/verl_data/two_policy_deepscaler_qwne1_7b_passrate_025_075/train.parquet
VAL_FILES=${REPO_ROOT}/verl_data/two_policy_aime2025_amc2023_eval/val.parquet
ABSTRACTION_PROMPT_TEMPLATE_PATH=${REPO_ROOT}/prompt_templates/sft_principle_generation.txt
SOLVER_PROMPT_TEMPLATE_PATH=${REPO_ROOT}/prompt_templates/hint_conditioned_problem_solving_rich_v1.txt
REWARD_FN_PATH=${REPO_ROOT}/verl/verl/utils/reward_score/deepscaler_math_reward_multibox_patched.py

for required_file in \
    "${TRAIN_FILES}" \
    "${VAL_FILES}" \
    "${ABSTRACTION_PROMPT_TEMPLATE_PATH}" \
    "${SOLVER_PROMPT_TEMPLATE_PATH}" \
    "${REWARD_FN_PATH}"; do
    if [[ ! -f "${required_file}" ]]; then
        echo "Required file not found: ${required_file}" >&2
        exit 1
    fi
done

if [[ ! -f "${ABSTRACTION_MODEL_PATH}/config.json" ]]; then
    echo "Abstraction model not found or incomplete: ${ABSTRACTION_MODEL_PATH}" >&2
    exit 1
fi

TRAINER_PROJECT_NAME=${TRAINER_PROJECT_NAME:-verl_two_policy}
TRAINER_EXPERIMENT_NAME=${TRAINER_EXPERIMENT_NAME:-qwen3_1.7b}
RUN_ROOT=${RUN_ROOT:-${REPO_ROOT}/two_policy_runs}
RUN_TAG=${RUN_TAG:-qwen3_1p7b_full_grpo_$(date -u +%Y%m%d_%H%M%S)}
RUN_DIR=${RUN_DIR:-${RUN_ROOT}/${TRAINER_EXPERIMENT_NAME}_${RUN_TAG}}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-${RUN_DIR}/checkpoints}
ROLLOUT_DIR=${ROLLOUT_DIR:-${RUN_DIR}/rollouts}
VALIDATION_DIR=${VALIDATION_DIR:-${RUN_DIR}/validation_rollouts}
WANDB_DIR=${WANDB_DIR:-${RUN_DIR}/wandb}

mkdir -p "${RUN_DIR}" "${CHECKPOINT_DIR}" "${ROLLOUT_DIR}" "${VALIDATION_DIR}" "${WANDB_DIR}"

export WANDB_DIR

echo "Two-policy full explicit launch:"
echo "  train_files=${TRAIN_FILES}"
echo "  val_files=${VAL_FILES}"
echo "  abstraction_model=${ABSTRACTION_MODEL_PATH}"
echo "  solver_model=Qwen/Qwen3-1.7B"
echo "  run_root=${RUN_ROOT}"
echo "  run_tag=${RUN_TAG}"
echo "  run_dir=${RUN_DIR}"
echo "  checkpoint_dir=${CHECKPOINT_DIR}"
echo "  rollout_dir=${ROLLOUT_DIR}"
echo "  validation_dir=${VALIDATION_DIR}"
echo "  wandb_dir=${WANDB_DIR}"

python3 -m recipe.two_policy.main_two_policy_ppo \
    hydra.searchpath=[file://${REPO_ROOT}/verl/verl/trainer/config] \
    algorithm.adv_estimator=grpo \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${VAL_FILES}" \
    data.train_batch_size=64 \
    data.val_batch_size=64 \
    data.max_prompt_length=4096 \
    data.max_response_length=8192 \
    actor_rollout_ref.model.path=Qwen/Qwen3-1.7B \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_liger=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    +actor_rollout_ref.actor.fsdp_config.grad_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.45 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.response_length=8192 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    abstraction_actor_rollout_ref.model.path="${ABSTRACTION_MODEL_PATH}" \
    abstraction_actor_rollout_ref.model.use_remove_padding=True \
    abstraction_actor_rollout_ref.model.enable_gradient_checkpointing=True \
    abstraction_actor_rollout_ref.model.use_liger=True \
    abstraction_actor_rollout_ref.actor.optim.lr=1e-6 \
    abstraction_actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    abstraction_actor_rollout_ref.actor.use_dynamic_bsz=True \
    abstraction_actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
    abstraction_actor_rollout_ref.actor.use_kl_loss=True \
    abstraction_actor_rollout_ref.actor.kl_loss_coef=0.001 \
    abstraction_actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    abstraction_actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    abstraction_actor_rollout_ref.actor.fsdp_config.param_offload=False \
    +abstraction_actor_rollout_ref.actor.fsdp_config.grad_offload=False \
    abstraction_actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    abstraction_actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    abstraction_actor_rollout_ref.rollout.name=vllm \
    abstraction_actor_rollout_ref.rollout.dtype=bfloat16 \
    abstraction_actor_rollout_ref.rollout.temperature=0.6 \
    abstraction_actor_rollout_ref.rollout.gpu_memory_utilization=0.25 \
    abstraction_actor_rollout_ref.rollout.n=1 \
    abstraction_actor_rollout_ref.rollout.response_length=2048 \
    abstraction_actor_rollout_ref.rollout.enforce_eager=True \
    abstraction_actor_rollout_ref.rollout.free_cache_engine=True \
    abstraction_actor_rollout_ref.rollout.load_format=safetensors \
    abstraction_actor_rollout_ref.ref.fsdp_config.param_offload=True \
    two_policy.abstraction_prompt_template_path="${ABSTRACTION_PROMPT_TEMPLATE_PATH}" \
    two_policy.solver_prompt_template_path="${SOLVER_PROMPT_TEMPLATE_PATH}" \
    two_policy.num_abstractions=4 \
    two_policy.num_solver_rollouts=4 \
    two_policy.validation_num_abstractions=4 \
    two_policy.validation_num_solver_rollouts=4 \
    two_policy.validation_solver_response_length=32768 \
    two_policy.decoupled_solver_schedule.enable=False \
    two_policy.decoupled_solver_schedule.solver_update_every_n_steps=1 \
    two_policy.decoupled_solver_schedule.non_update_solver_rollouts=1 \
    two_policy.decoupled_solver_schedule.non_update_solver_temperature=0.0 \
    two_policy.decoupled_solver_schedule.non_update_solver_top_p=1.0 \
    two_policy.decoupled_solver_schedule.non_update_solver_top_k=-1 \
    two_policy.vm_reward.enable=False \
    two_policy.vm_reward.transform=logit \
    two_policy.vm_reward.solver_weight=0.1 \
    two_policy.vm_reward.abstraction_weight=0.1 \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="${TRAINER_PROJECT_NAME}" \
    trainer.experiment_name="${TRAINER_EXPERIMENT_NAME}" \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.rollout_data_dir="${ROLLOUT_DIR}" \
    trainer.validation_data_dir="${VALIDATION_DIR}" \
    trainer.default_local_dir="${CHECKPOINT_DIR}" \
    trainer.save_freq=100 \
    trainer.test_freq=20 \
    trainer.total_epochs=10 \
    trainer.resume_mode=auto \
    trainer.val_before_train=True \
    trainer.max_actor_ckpt_to_keep=1 \
    reward.custom_reward_function.path="${REWARD_FN_PATH}" \
    reward.custom_reward_function.name=compute_score \
    reward.reward_manager.name=naive \
    ray_kwargs.ray_init.num_cpus=8 \
    "$@"
