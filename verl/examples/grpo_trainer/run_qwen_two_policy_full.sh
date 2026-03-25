#!/bin/bash
set -euo pipefail

cd /workspace/action_abstraction/verl

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING=1

ABSTRACTION_MODEL_PATH=${ABSTRACTION_MODEL_PATH:-/workspace/action_abstraction/merged_models/qwen3_1_7b_principle_generator_ckpt1736}
SOLVER_MODEL_PATH=${SOLVER_MODEL_PATH:-Qwen/Qwen3-1.7B}
TRAIN_FILES=${TRAIN_FILES:-/workspace/action_abstraction/verl_data/two_policy_deepscaler_qwne1_7b_passrate_025_075/train.parquet}
VAL_FILES=${VAL_FILES:-/workspace/action_abstraction/verl_data/two_policy_aime2025_amc2023_eval/val.parquet}
ABSTRACTION_MAX_RESP_LEN=${ABSTRACTION_MAX_RESP_LEN:-1024}
SOLVER_MAX_RESP_LEN=${SOLVER_MAX_RESP_LEN:-8192}
VALIDATION_SOLVER_MAX_RESP_LEN=${VALIDATION_SOLVER_MAX_RESP_LEN:-32768}
MAX_RESP_LEN=$(( ABSTRACTION_MAX_RESP_LEN > SOLVER_MAX_RESP_LEN ? ABSTRACTION_MAX_RESP_LEN : SOLVER_MAX_RESP_LEN ))
RUN_TAG=${RUN_TAG:-$(date -u +%Y%m%d_%H%M%S)}
RUN_ROOT=${RUN_ROOT:-/tmp/action_abstraction/two_policy_runs}
RUN_DIR=${RUN_DIR:-${RUN_ROOT}/full_grpo_${RUN_TAG}}
ROLLOUT_DIR=${ROLLOUT_DIR:-${RUN_DIR}/rollouts}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-${RUN_DIR}/checkpoints}
VALIDATION_DIR=${VALIDATION_DIR:-${RUN_DIR}/validation_rollouts}

mkdir -p "${RUN_DIR}" "${ROLLOUT_DIR}" "${CHECKPOINT_DIR}" "${VALIDATION_DIR}"

args=(
  hydra.searchpath=[file:///workspace/action_abstraction/verl/verl/trainer/config]
  algorithm.adv_estimator=grpo
  data.train_files="${TRAIN_FILES}"
  data.val_files="${VAL_FILES}"
  data.train_batch_size=64
  data.val_batch_size=64
  data.max_prompt_length=4096
  data.max_response_length="${MAX_RESP_LEN}"
  actor_rollout_ref.model.path="${SOLVER_MODEL_PATH}"
  # actor_rollout_ref.model.lora_rank=32
  # actor_rollout_ref.model.lora_alpha=32
  # actor_rollout_ref.model.target_modules=all-linear
  actor_rollout_ref.model.use_remove_padding=True
  actor_rollout_ref.model.enable_gradient_checkpointing=True
  actor_rollout_ref.model.use_liger=True
  actor_rollout_ref.actor.optim.lr=1e-6
  actor_rollout_ref.actor.ppo_mini_batch_size=32
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
  actor_rollout_ref.rollout.gpu_memory_utilization=0.7
  actor_rollout_ref.rollout.n=1
  actor_rollout_ref.rollout.response_length="${SOLVER_MAX_RESP_LEN}"
  actor_rollout_ref.rollout.enforce_eager=False
  actor_rollout_ref.rollout.free_cache_engine=True
  actor_rollout_ref.rollout.load_format=safetensors
  actor_rollout_ref.ref.fsdp_config.param_offload=True
  abstraction_actor_rollout_ref.model.path="${ABSTRACTION_MODEL_PATH}"
  # abstraction_actor_rollout_ref.model.lora_rank=32
  # abstraction_actor_rollout_ref.model.lora_alpha=32
  # abstraction_actor_rollout_ref.model.target_modules=all-linear
  abstraction_actor_rollout_ref.model.use_remove_padding=True
  abstraction_actor_rollout_ref.model.enable_gradient_checkpointing=True
  abstraction_actor_rollout_ref.model.use_liger=True
  abstraction_actor_rollout_ref.actor.optim.lr=1e-6
  abstraction_actor_rollout_ref.actor.ppo_mini_batch_size=32
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
  abstraction_actor_rollout_ref.rollout.gpu_memory_utilization=0.7
  abstraction_actor_rollout_ref.rollout.n=1
  abstraction_actor_rollout_ref.rollout.response_length="${ABSTRACTION_MAX_RESP_LEN}"
  abstraction_actor_rollout_ref.rollout.enforce_eager=False
  abstraction_actor_rollout_ref.rollout.free_cache_engine=True
  abstraction_actor_rollout_ref.rollout.load_format=safetensors
  abstraction_actor_rollout_ref.ref.fsdp_config.param_offload=True
  two_policy.num_abstractions=4
  two_policy.num_solver_rollouts=4
  two_policy.validation_num_abstractions=4
  two_policy.validation_num_solver_rollouts=4
  two_policy.validation_solver_response_length="${VALIDATION_SOLVER_MAX_RESP_LEN}"
  trainer.critic_warmup=0
  trainer.logger=[console,wandb]
  trainer.project_name=verl_two_policy
  trainer.experiment_name=qwen3_1.7b
  trainer.n_gpus_per_node=1
  trainer.nnodes=1
  trainer.rollout_data_dir="${ROLLOUT_DIR}"
  trainer.validation_data_dir="${VALIDATION_DIR}"
  trainer.default_local_dir="${CHECKPOINT_DIR}"
  trainer.save_freq=100
  trainer.test_freq=20
  trainer.total_epochs=10
  trainer.resume_mode=auto
  trainer.max_actor_ckpt_to_keep=1
  reward.custom_reward_function.path=/workspace/action_abstraction/verl/verl/utils/reward_score/deepscaler_math_reward_multibox_patched.py
  reward.custom_reward_function.name=compute_score
  reward.reward_manager.name=naive
  ray_kwargs.ray_init.num_cpus=64
)

python3 -m recipe.two_policy.main_two_policy_ppo "${args[@]}" "$@"
