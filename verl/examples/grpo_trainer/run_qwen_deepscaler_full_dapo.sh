#!/bin/bash
set -eu

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING=1


python3 -m recipe.dapo.main_dapo \
    hydra.searchpath="[file:///workspace/action_abstraction/verl/verl/trainer/config]" \
    algorithm.adv_estimator=grpo \
    data.train_files=/workspace/action_abstraction/verl_data/sft_dataset_no_rl_partial_05_less1_traintest_concat/train.parquet \
    data.val_files=/workspace/action_abstraction/verl_data/aime_amc_with_abstraction/test.parquet \
    data.train_batch_size=64 \
    data.val_batch_size=70 \
    data.max_prompt_length=3420 \
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
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.rollout_data_dir=/workspace/action_abstraction/verl_outputs2 \
    trainer.project_name='verl_deepscaler_easy' \
    trainer.experiment_name='verl_deepscaler_easy_1.7B' \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=2000 \
    trainer.test_freq=20 \
    trainer.total_epochs=10 \
    reward.custom_reward_function.path=/workspace/action_abstraction/verl/verl/utils/reward_score/deepscaler_math_reward_multibox_patched.py \
    reward.custom_reward_function.name=compute_score \
    reward.reward_manager.name=naive \
    ray_kwargs.ray_init.num_cpus=180 \
    trainer.default_local_dir=/checkpoints \
    trainer.resume_mode=auto \
    trainer.max_actor_ckpt_to_keep=1 \
    trainer.max_critic_ckpt_to_keep=1 \
    algorithm.filter_groups.enable=True \
    algorithm.filter_groups.metric=acc \
    algorithm.filter_groups.max_num_gen_batches=4 \
    "$@"
