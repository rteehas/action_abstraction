#!/bin/bash
set -eu

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING=1


python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/scratch/rst306/action_abstractions/action_abstraction/verl_data/sft_dataset_no_rl_partial_05_less1_traintest_overfitting/train.parquet \
    data.val_files=/scratch/rst306/action_abstractions/action_abstraction/verl_data/sft_dataset_no_rl_partial_05_less1_traintest_overfitting/test.parquet \
    data.train_batch_size=2 \
    data.val_batch_size=2 \
    data.max_prompt_length=1024 \
    data.max_response_length=8192 \
    actor_rollout_ref.model.path=Qwen/Qwen3-1.7B \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_liger=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=10000 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    +actor_rollout_ref.actor.fsdp_config.grad_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.1 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.rollout_data_dir=/scratch/rst306/action_abstractions/action_abstraction/verl_temp_outputs \
    trainer.project_name='verl_deepscaler' \
    trainer.experiment_name='verl_deepscaler_overfitting_1.7B' \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=10000 \
    trainer.test_freq=10000 \
    trainer.total_epochs=100000000000 \
    reward.custom_reward_function.path=/scratch/rst306/action_abstractions/action_abstraction/verl/verl/utils/reward_score/deepscaler_math_reward.py \
    reward.custom_reward_function.name=compute_score \
    reward.reward_manager.name=naive
    "$@"
