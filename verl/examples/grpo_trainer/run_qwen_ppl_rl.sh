#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}

if [[ -z "${CONDA_DEFAULT_ENV:-}" ]] && [[ -f /workspace/miniconda3/etc/profile.d/conda.sh ]]; then
  source /workspace/miniconda3/etc/profile.d/conda.sh
  conda activate abstraction
fi

RUN_TAG=${RUN_TAG:-$(date -u +%Y%m%d_%H%M%S)}
TRAIN_FILES=${TRAIN_FILES:-/tmp/action_abstraction/ppl_rl_tiny_overfit/train.parquet}
VAL_FILES=${VAL_FILES:-/tmp/action_abstraction/ppl_rl_tiny_overfit/val.parquet}
ACTOR_MODEL_PATH=${ACTOR_MODEL_PATH:-${REPO_ROOT}/merged_models/qwen3_1_7b_principle_generator_ckpt1736}
PPL_MODEL_PATH=${PPL_MODEL_PATH:-Qwen/Qwen3-1.7B}
SOLVER_PROMPT_TEMPLATE_PATH=${SOLVER_PROMPT_TEMPLATE_PATH:-${REPO_ROOT}/prompt_templates/hint_conditioned_problem_solving_rich_v1.txt}
RUN_ROOT=${RUN_ROOT:-/tmp/action_abstraction/ppl_rl_runs}
RUN_DIR=${RUN_DIR:-${RUN_ROOT}/run_${RUN_TAG}}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-${RUN_DIR}/checkpoints}
ROLLOUT_DIR=${ROLLOUT_DIR:-${RUN_DIR}/rollouts}
HYDRA_RUN_DIR=${HYDRA_RUN_DIR:-${RUN_DIR}/hydra}
RAY_NUM_CPUS=${RAY_NUM_CPUS:-8}
DATA_DATALOADER_NUM_WORKERS=${DATA_DATALOADER_NUM_WORKERS:-0}
MAX_PROMPT_LEN=${MAX_PROMPT_LEN:-4096}
MAX_RESPONSE_LEN=${MAX_RESPONSE_LEN:-1024}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-1}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-2}
LR=${LR:-1e-5}
ROLLOUT_N=${ROLLOUT_N:-4}
TEMPERATURE=${TEMPERATURE:-1.0}
TOP_P=${TOP_P:-0.95}
TOP_K=${TOP_K:-20}
LORA_RANK=${LORA_RANK:-4}
LORA_ALPHA=${LORA_ALPHA:-8}
ACTOR_GPU_MEM_UTIL=${ACTOR_GPU_MEM_UTIL:-0.45}
REWARD_GPU_MEM_UTIL=${REWARD_GPU_MEM_UTIL:-0.20}
UNCONDITIONAL_LAMBDA=${UNCONDITIONAL_LAMBDA:-0.1}
REWARD_CLIP=${REWARD_CLIP:-True}
POSITIVE_IMPROVEMENT_ONLY=${POSITIVE_IMPROVEMENT_ONLY:-True}
AGENT_NUM_WORKERS=${AGENT_NUM_WORKERS:-1}
TRAINER_PROJECT_NAME=${TRAINER_PROJECT_NAME:-ppl_rl_verl}
TRAINER_EXPERIMENT_NAME=${TRAINER_EXPERIMENT_NAME:-qwen3_1.7b_ppl_rl_${RUN_TAG}}
TRAINER_TOTAL_EPOCHS=${TRAINER_TOTAL_EPOCHS:-2}
TRAINER_SAVE_FREQ=${TRAINER_SAVE_FREQ:-10}
TRAINER_TEST_FREQ=${TRAINER_TEST_FREQ:-10}

mkdir -p "${RUN_DIR}" "${CHECKPOINT_DIR}" "${ROLLOUT_DIR}"
mkdir -p "${HYDRA_RUN_DIR}"

ray stop --force >/dev/null 2>&1 || true

cd "${REPO_ROOT}/verl"

args=(
  hydra.searchpath=[file://${REPO_ROOT}/verl/verl/trainer/config]
  hydra.run.dir="${HYDRA_RUN_DIR}"
  hydra.job.chdir=False
  data.train_files="${TRAIN_FILES}"
  data.val_files="${VAL_FILES}"
  data.train_batch_size="${TRAIN_BATCH_SIZE}"
  data.val_batch_size="${VAL_BATCH_SIZE}"
  data.dataloader_num_workers="${DATA_DATALOADER_NUM_WORKERS}"
  data.max_prompt_length="${MAX_PROMPT_LEN}"
  data.max_response_length="${MAX_RESPONSE_LEN}"
  actor_rollout_ref.model.path="${ACTOR_MODEL_PATH}"
  actor_rollout_ref.model.lora_rank="${LORA_RANK}"
  actor_rollout_ref.model.lora_alpha="${LORA_ALPHA}"
  actor_rollout_ref.model.target_modules=all-linear
  actor_rollout_ref.model.use_remove_padding=True
  actor_rollout_ref.model.enable_gradient_checkpointing=True
  actor_rollout_ref.actor.optim.lr="${LR}"
  actor_rollout_ref.actor.ppo_mini_batch_size="${TRAIN_BATCH_SIZE}"
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
  actor_rollout_ref.rollout.temperature="${TEMPERATURE}"
  actor_rollout_ref.rollout.top_p="${TOP_P}"
  actor_rollout_ref.rollout.top_k="${TOP_K}"
  actor_rollout_ref.rollout.gpu_memory_utilization="${ACTOR_GPU_MEM_UTIL}"
  actor_rollout_ref.rollout.n="${ROLLOUT_N}"
  actor_rollout_ref.rollout.response_length="${MAX_RESPONSE_LEN}"
  actor_rollout_ref.rollout.agent.num_workers="${AGENT_NUM_WORKERS}"
  actor_rollout_ref.rollout.enforce_eager=True
  actor_rollout_ref.rollout.free_cache_engine=True
  actor_rollout_ref.rollout.load_format=safetensors
  actor_rollout_ref.ref.fsdp_config.param_offload=True
  reward.custom_reward_function.path="${REPO_ROOT}/verl/recipe/ppl_rl/reward_fn_ppl.py"
  reward.custom_reward_function.name=compute_score
  reward.custom_reward_function.reward_kwargs.hint_template_path="${SOLVER_PROMPT_TEMPLATE_PATH}"
  reward.custom_reward_function.reward_kwargs.reward_model_name="${PPL_MODEL_PATH}"
  reward.custom_reward_function.reward_kwargs.unconditional_lambda="${UNCONDITIONAL_LAMBDA}"
  reward.custom_reward_function.reward_kwargs.clip_reward="${REWARD_CLIP}"
  reward.custom_reward_function.reward_kwargs.positive_improvement_only="${POSITIVE_IMPROVEMENT_ONLY}"
  reward.num_workers=1
  reward.reward_manager.name=naive
  reward.reward_model.enable=True
  reward.reward_model.enable_resource_pool=False
  reward.reward_model.model_path="${PPL_MODEL_PATH}"
  reward.reward_model.rollout.name=vllm
  reward.reward_model.rollout.dtype=bfloat16
  reward.reward_model.rollout.tensor_model_parallel_size=1
  reward.reward_model.rollout.gpu_memory_utilization="${REWARD_GPU_MEM_UTIL}"
  reward.reward_model.rollout.max_num_batched_tokens=4096
  reward.reward_model.rollout.max_model_len=4096
  reward.reward_model.rollout.free_cache_engine=True
  reward.reward_model.rollout.skip_tokenizer_init=False
  reward.reward_model.rollout.prompt_length=4096
  reward.reward_model.rollout.response_length=1
  trainer.logger=[console,wandb]
  trainer.project_name="${TRAINER_PROJECT_NAME}"
  trainer.experiment_name="${TRAINER_EXPERIMENT_NAME}"
  trainer.n_gpus_per_node=1
  trainer.nnodes=1
  trainer.rollout_data_dir="${ROLLOUT_DIR}"
  trainer.default_local_dir="${CHECKPOINT_DIR}"
  trainer.save_freq="${TRAINER_SAVE_FREQ}"
  trainer.test_freq="${TRAINER_TEST_FREQ}"
  trainer.total_epochs="${TRAINER_TOTAL_EPOCHS}"
  trainer.max_actor_ckpt_to_keep=4
  algorithm.use_kl_in_reward=False
  algorithm.adv_estimator=grpo
  ray_kwargs.ray_init.num_cpus="${RAY_NUM_CPUS}"
)

python -m recipe.ppl_rl.main_ppl_rl "${args[@]}" "$@"
