set -x

export VLLM_ATTENTION_BACKEND=FLASH_ATTN

# Default values
MODEL_PATH="/checkpoints/global_step_200"
# Possible values: aime, amc, math, minerva, olympiad_bench

# Echo the values for verification
echo "Model Path: ${MODEL_PATH}"
echo "Datasets: ${DATATYPES[@]}"
echo "Output Directory: ${OUTPUT_DIR}"

# Loop through all datatypes
python3 -m verl.trainer.main_generation \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=1 \
    data.path=/workspace/action_abstraction/verl_data/aime_amc_with_abstraction/test.parquet \
    data.output_path=verl_eval/aime_amc_test.parquet \
    data.n_samples=16 \
    data.batch_size=140 \
    model.path=${MODEL_PATH} \
    rollout.temperature=0.6 \
    rollout.response_length=32768 \
    rollout.top_k=-1 \
    rollout.top_p=0.95 \
    rollout.gpu_memory_utilization=0.9 \
    rollout.tensor_model_parallel_size=1 \
    ray_kwargs.ray_init.num_cpus=180 \
    reward.custom_reward_function.path=/workspace/action_abstraction/verl/verl/utils/reward_score/deepscaler_math_reward_multibox_patched.py \
    reward.custom_reward_function.name=compute_score \
    reward.reward_manager.name=naive 

