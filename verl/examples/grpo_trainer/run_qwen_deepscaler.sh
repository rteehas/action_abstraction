set -x

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    trainer.val_before_train=False \
    data.train_files=/scratch/rst306/action_abstractions/action_abstraction/verl_data/sft_dataset_no_rl_partial_05_less1_traintest_overfitting/train.parquet \
    data.val_files=/scratch/rst306/action_abstractions/action_abstraction/verl_data/sft_dataset_no_rl_partial_05_less1_traintest_overfitting/test.parquet \
    data.train_batch_size=2 \
    data.max_prompt_length=1024 \
    data.max_response_length=8000 \
    data.filter_overlong_prompts=False \
    data.truncation='error' \
    data.shuffle=False \
    actor_rollout_ref.model.path=Qwen/Qwen3-0.6B \
    actor_rollout_ref.model.lora_rank=64 \
    actor_rollout_ref.model.lora_alpha=32 \
    actor_rollout_ref.actor.optim.lr=3e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=40 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.1 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=40 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='verl_grpo_example_gsm8k' \
    trainer.experiment_name='qwen2.5_3b_grpo_lora' \
    ray_kwargs.ray_init.num_cpus=8 \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=1000 \
    trainer.test_freq=1000 \
    trainer.total_epochs=10000000000000 \
    reward.custom_reward_function.path=/scratch/rst306/action_abstractions/action_abstraction/verl/verl/utils/reward_score/deepscaler_math_reward.py \
    reward.custom_reward_function.name=compute_score \
    reward.reward_manager.name=naive
    "$@"

# YOU MAY NEED TO CHANGE:
# data.train_files=/scratch/rst306/action_abstractions/action_abstraction/verl_data/gsm8k/train.parquet
# data.val_files=/scratch/rst306/action_abstractions/action_abstraction/verl_data/gsm8k/test.parquet
# trainer.n_gpus_per_node=2

# actor_rollout_ref.actor.ppo_mini_batch_size=256
# data.train_batch_size=1024
# trainer.n_gpus_per_node=8
# actor_rollout_ref.model.use_shm=True