#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}

cd "${REPO_ROOT}/verl"

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1

# Fully expanded version of:
#   run_qwen_two_policy_deepscaler_sample128_h200_balanced.sh
#   -> run_qwen_two_policy_deepscaler_sample128_debug.sh
#   -> run_qwen_two_policy_full_dapo.sh
#
# Every default used by this launcher is declared below. There is no profile
# logic, alias handling, or preset resolution inside this file.

# Paths
ABSTRACTION_MODEL_PATH=${ABSTRACTION_MODEL_PATH:-${REPO_ROOT}/merged_models/qwen3_1_7b_principle_generator_ckpt1736}
SOLVER_MODEL_PATH=${SOLVER_MODEL_PATH:-Qwen/Qwen3-1.7B}
ABSTRACTION_PROMPT_TEMPLATE_PATH=${ABSTRACTION_PROMPT_TEMPLATE_PATH:-${REPO_ROOT}/prompt_templates/sft_principle_generation.txt}
SOLVER_PROMPT_TEMPLATE_PATH=${SOLVER_PROMPT_TEMPLATE_PATH:-${REPO_ROOT}/prompt_templates/hint_conditioned_problem_solving_rich_v1.txt}
DATASET_DIR=${DATASET_DIR:-${REPO_ROOT}/verl_data/two_policy_deepscaler_qwne1_7b_passrate_025_075_sample128_seed0}
TRAIN_FILES=${TRAIN_FILES:-${DATASET_DIR}/train.parquet}
DEFAULT_VAL_FILES="${DATASET_DIR}/val.parquet"
if [[ -z "${VAL_FILES:-}" ]]; then
  if [[ -f "${DEFAULT_VAL_FILES}" ]]; then
    VAL_FILES="${DEFAULT_VAL_FILES}"
  else
    VAL_FILES="${TRAIN_FILES}"
  fi
fi

# Run directories
RUN_ROOT=${RUN_ROOT:-${REPO_ROOT}/two_policy_runs}
RUN_TAG=${RUN_TAG:-sample128_h200_balanced_bs8_abs4_sol8_$(date -u +%Y%m%d_%H%M%S)}
RUN_DIR=${RUN_DIR:-${RUN_ROOT}/${RUN_TAG}}
ROLLOUT_DIR=${ROLLOUT_DIR:-${RUN_DIR}/rollouts}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-${RUN_DIR}/checkpoints}
VALIDATION_DIR=${VALIDATION_DIR:-${RUN_DIR}/validation_rollouts}

# Data and rollout shape
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-4096}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-8}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-8}
TWO_POLICY_NUM_ABSTRACTIONS=${TWO_POLICY_NUM_ABSTRACTIONS:-4}
TWO_POLICY_NUM_SOLVER_ROLLOUTS=${TWO_POLICY_NUM_SOLVER_ROLLOUTS:-8}
TWO_POLICY_VALIDATION_NUM_ABSTRACTIONS=${TWO_POLICY_VALIDATION_NUM_ABSTRACTIONS:-1}
TWO_POLICY_VALIDATION_NUM_SOLVER_ROLLOUTS=${TWO_POLICY_VALIDATION_NUM_SOLVER_ROLLOUTS:-1}
ABSTRACTION_MAX_RESP_LEN=${ABSTRACTION_MAX_RESP_LEN:-2048}
SOLVER_MAX_RESP_LEN=${SOLVER_MAX_RESP_LEN:-8192}
VALIDATION_SOLVER_MAX_RESP_LEN=${VALIDATION_SOLVER_MAX_RESP_LEN:-8192}
MAX_RESP_LEN=$(( ABSTRACTION_MAX_RESP_LEN > SOLVER_MAX_RESP_LEN ? ABSTRACTION_MAX_RESP_LEN : SOLVER_MAX_RESP_LEN ))

# H200-balanced rollout settings
SOLVER_GPU_MEM_UTIL=${SOLVER_GPU_MEM_UTIL:-0.50}
ABSTRACTION_GPU_MEM_UTIL=${ABSTRACTION_GPU_MEM_UTIL:-0.28}
ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-144}
SOLVER_MAX_BATCHED_TOKENS=${SOLVER_MAX_BATCHED_TOKENS:-10240}
ABSTRACTION_MAX_BATCHED_TOKENS=${ABSTRACTION_MAX_BATCHED_TOKENS:-2560}
SOLVER_ROLLOUT_TEMPERATURE=${SOLVER_ROLLOUT_TEMPERATURE:-0.6}
SOLVER_ROLLOUT_TOP_P=${SOLVER_ROLLOUT_TOP_P:-0.95}
SOLVER_ROLLOUT_TOP_K=${SOLVER_ROLLOUT_TOP_K:-20}
ABSTRACTION_ROLLOUT_TEMPERATURE=${ABSTRACTION_ROLLOUT_TEMPERATURE:-0.6}
ABSTRACTION_ROLLOUT_TOP_P=${ABSTRACTION_ROLLOUT_TOP_P:-0.95}
ABSTRACTION_ROLLOUT_TOP_K=${ABSTRACTION_ROLLOUT_TOP_K:-20}
ROLLOUT_ENFORCE_EAGER=${ROLLOUT_ENFORCE_EAGER:-True}
ROLLOUT_FREE_CACHE_ENGINE=${ROLLOUT_FREE_CACHE_ENGINE:-True}

# Actor and optimizer settings
ACTOR_LR=${ACTOR_LR:-1e-6}
ACTOR_PPO_MINI_BATCH_SIZE=${ACTOR_PPO_MINI_BATCH_SIZE:-32}
ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU=${ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU:-32768}
ACTOR_PARAM_OFFLOAD=${ACTOR_PARAM_OFFLOAD:-True}
ACTOR_OPTIMIZER_OFFLOAD=${ACTOR_OPTIMIZER_OFFLOAD:-True}
USE_KL_LOSS=${USE_KL_LOSS:-True}
ACTOR_KL_LOSS_COEF=${ACTOR_KL_LOSS_COEF:-0.001}
ACTOR_KL_LOSS_TYPE=${ACTOR_KL_LOSS_TYPE:-low_var_kl}

# Debug-style trainer settings
TRAINER_SAVE_FREQ=${TRAINER_SAVE_FREQ:-99999}
TRAINER_TEST_FREQ=${TRAINER_TEST_FREQ:-0}
TRAINER_TOTAL_EPOCHS=${TRAINER_TOTAL_EPOCHS:-3}
TRAINER_VAL_BEFORE_TRAIN=${TRAINER_VAL_BEFORE_TRAIN:-False}
TRAINER_PROJECT_NAME=${TRAINER_PROJECT_NAME:-verl_two_policy_debug}
TRAINER_EXPERIMENT_NAME=${TRAINER_EXPERIMENT_NAME:-deepscaler_sample128_h200_balanced_qwen3_1p7b}
TRAINER_RESUME_MODE=${TRAINER_RESUME_MODE:-disable}
TRAINER_MAX_ACTOR_CKPT_TO_KEEP=${TRAINER_MAX_ACTOR_CKPT_TO_KEEP:-1}

# Ray, workers, and filtering
RAY_NUM_CPUS=${RAY_NUM_CPUS:-8}
RAY_TIMELINE_JSON_FILE=${RAY_TIMELINE_JSON_FILE:-}
DATA_DATALOADER_NUM_WORKERS=${DATA_DATALOADER_NUM_WORKERS:-0}
REWARD_NUM_WORKERS=${REWARD_NUM_WORKERS:-1}
SOLVER_AGENT_NUM_WORKERS=${SOLVER_AGENT_NUM_WORKERS:-1}
ABSTRACTION_AGENT_NUM_WORKERS=${ABSTRACTION_AGENT_NUM_WORKERS:-1}
FILTER_GROUPS_ENABLE=${FILTER_GROUPS_ENABLE:-False}
FILTER_GROUPS_METRIC=${FILTER_GROUPS_METRIC:-acc}
FILTER_GROUPS_MAX_NUM_GEN_BATCHES=${FILTER_GROUPS_MAX_NUM_GEN_BATCHES:-1}
TWO_POLICY_FILTER_SOLVER_GROUPS=${TWO_POLICY_FILTER_SOLVER_GROUPS:-False}

# Two-policy schedule and VM reward settings
TWO_POLICY_DECOUPLED_SOLVER_SCHEDULE_ENABLE=${TWO_POLICY_DECOUPLED_SOLVER_SCHEDULE_ENABLE:-False}
TWO_POLICY_SOLVER_UPDATE_EVERY_N_STEPS=${TWO_POLICY_SOLVER_UPDATE_EVERY_N_STEPS:-1}
TWO_POLICY_NON_UPDATE_SOLVER_ROLLOUTS=${TWO_POLICY_NON_UPDATE_SOLVER_ROLLOUTS:-1}
TWO_POLICY_NON_UPDATE_SOLVER_TEMPERATURE=${TWO_POLICY_NON_UPDATE_SOLVER_TEMPERATURE:-0.0}
TWO_POLICY_NON_UPDATE_SOLVER_TOP_P=${TWO_POLICY_NON_UPDATE_SOLVER_TOP_P:-1.0}
TWO_POLICY_NON_UPDATE_SOLVER_TOP_K=${TWO_POLICY_NON_UPDATE_SOLVER_TOP_K:--1}
TWO_POLICY_VM_REWARD_ENABLE=${TWO_POLICY_VM_REWARD_ENABLE:-False}
TWO_POLICY_VM_REWARD_TRANSFORM=${TWO_POLICY_VM_REWARD_TRANSFORM:-logit}
TWO_POLICY_VM_REWARD_SOLVER_WEIGHT=${TWO_POLICY_VM_REWARD_SOLVER_WEIGHT:-0.1}
TWO_POLICY_VM_REWARD_ABSTRACTION_WEIGHT=${TWO_POLICY_VM_REWARD_ABSTRACTION_WEIGHT:-0.1}

# Optional profiler settings
GLOBAL_PROFILER_TOOL=${GLOBAL_PROFILER_TOOL:-}
GLOBAL_PROFILER_STEPS=${GLOBAL_PROFILER_STEPS:-[1]}
GLOBAL_PROFILER_CONTINUOUS_STEPS=${GLOBAL_PROFILER_CONTINUOUS_STEPS:-False}
GLOBAL_PROFILER_SAVE_PATH=${GLOBAL_PROFILER_SAVE_PATH:-${RUN_DIR}/profile}
DEFAULT_ACTOR_PROFILER_ENABLE=False
if [[ -n "${GLOBAL_PROFILER_TOOL}" ]]; then
  DEFAULT_ACTOR_PROFILER_ENABLE=True
fi
ACTOR_PROFILER_ENABLE=${ACTOR_PROFILER_ENABLE:-${DEFAULT_ACTOR_PROFILER_ENABLE}}
ACTOR_PROFILER_ALL_RANKS=${ACTOR_PROFILER_ALL_RANKS:-False}
ACTOR_PROFILER_RANKS=${ACTOR_PROFILER_RANKS:-[]}
ACTOR_TORCH_PROFILER_CONTENTS=${ACTOR_TORCH_PROFILER_CONTENTS:-[cuda]}
ABSTRACTION_ACTOR_PROFILER_ENABLE=${ABSTRACTION_ACTOR_PROFILER_ENABLE:-${ACTOR_PROFILER_ENABLE}}
ABSTRACTION_ACTOR_PROFILER_ALL_RANKS=${ABSTRACTION_ACTOR_PROFILER_ALL_RANKS:-${ACTOR_PROFILER_ALL_RANKS}}
ABSTRACTION_ACTOR_PROFILER_RANKS=${ABSTRACTION_ACTOR_PROFILER_RANKS:-${ACTOR_PROFILER_RANKS}}
ABSTRACTION_TORCH_PROFILER_CONTENTS=${ABSTRACTION_TORCH_PROFILER_CONTENTS:-${ACTOR_TORCH_PROFILER_CONTENTS}}
TORCH_MEMORY_TRACE_ALLOC_MAX_ENTRIES=${TORCH_MEMORY_TRACE_ALLOC_MAX_ENTRIES:-100000}
TORCH_MEMORY_STACK_DEPTH=${TORCH_MEMORY_STACK_DEPTH:-32}

if [[ ! -f "${TRAIN_FILES}" ]]; then
  echo "Train parquet not found: ${TRAIN_FILES}" >&2
  exit 1
fi

mkdir -p "${RUN_DIR}" "${ROLLOUT_DIR}" "${CHECKPOINT_DIR}" "${VALIDATION_DIR}"

echo "Sample128 H200-balanced explicit launch:"
echo "  train_files=${TRAIN_FILES}"
echo "  val_files=${VAL_FILES}"
echo "  run_dir=${RUN_DIR}"
echo "  train_batch_size=${TRAIN_BATCH_SIZE}"
echo "  num_abstractions=${TWO_POLICY_NUM_ABSTRACTIONS}"
echo "  num_solver_rollouts=${TWO_POLICY_NUM_SOLVER_ROLLOUTS}"
echo "  validation_num_abstractions=${TWO_POLICY_VALIDATION_NUM_ABSTRACTIONS}"
echo "  validation_num_solver_rollouts=${TWO_POLICY_VALIDATION_NUM_SOLVER_ROLLOUTS}"
echo "  solver_gpu_mem_util=${SOLVER_GPU_MEM_UTIL}"
echo "  abstraction_gpu_mem_util=${ABSTRACTION_GPU_MEM_UTIL}"
echo "  rollout_max_num_seqs=${ROLLOUT_MAX_NUM_SEQS}"
echo "  solver_max_batched_tokens=${SOLVER_MAX_BATCHED_TOKENS}"
echo "  abstraction_max_batched_tokens=${ABSTRACTION_MAX_BATCHED_TOKENS}"
echo "  actor_ppo_max_token_len_per_gpu=${ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU}"
echo "  trainer_resume_mode=${TRAINER_RESUME_MODE}"

args=(
  hydra.searchpath=[file://${REPO_ROOT}/verl/verl/trainer/config]
  algorithm.adv_estimator=grpo
  data.train_files="${TRAIN_FILES}"
  data.val_files="${VAL_FILES}"
  data.train_batch_size="${TRAIN_BATCH_SIZE}"
  data.val_batch_size="${VAL_BATCH_SIZE}"
  data.max_prompt_length="${MAX_PROMPT_LENGTH}"
  data.max_response_length="${MAX_RESP_LEN}"
  data.dataloader_num_workers="${DATA_DATALOADER_NUM_WORKERS}"
  actor_rollout_ref.model.path="${SOLVER_MODEL_PATH}"
  actor_rollout_ref.model.lora_rank=32
  actor_rollout_ref.model.lora_alpha=32
  actor_rollout_ref.model.target_modules=all-linear
  actor_rollout_ref.model.use_remove_padding=True
  actor_rollout_ref.model.enable_gradient_checkpointing=True
  actor_rollout_ref.model.use_liger=True
  actor_rollout_ref.actor.optim.lr="${ACTOR_LR}"
  actor_rollout_ref.actor.ppo_mini_batch_size="${ACTOR_PPO_MINI_BATCH_SIZE}"
  actor_rollout_ref.actor.use_dynamic_bsz=True
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU}"
  actor_rollout_ref.actor.use_kl_loss="${USE_KL_LOSS}"
  actor_rollout_ref.actor.kl_loss_coef="${ACTOR_KL_LOSS_COEF}"
  actor_rollout_ref.actor.kl_loss_type="${ACTOR_KL_LOSS_TYPE}"
  actor_rollout_ref.actor.ulysses_sequence_parallel_size=1
  actor_rollout_ref.actor.fsdp_config.param_offload="${ACTOR_PARAM_OFFLOAD}"
  +actor_rollout_ref.actor.fsdp_config.grad_offload=False
  actor_rollout_ref.actor.fsdp_config.optimizer_offload="${ACTOR_OPTIMIZER_OFFLOAD}"
  actor_rollout_ref.rollout.tensor_model_parallel_size=1
  actor_rollout_ref.rollout.name=vllm
  actor_rollout_ref.rollout.dtype=bfloat16
  actor_rollout_ref.rollout.temperature="${SOLVER_ROLLOUT_TEMPERATURE}"
  actor_rollout_ref.rollout.top_p="${SOLVER_ROLLOUT_TOP_P}"
  actor_rollout_ref.rollout.top_k="${SOLVER_ROLLOUT_TOP_K}"
  actor_rollout_ref.rollout.gpu_memory_utilization="${SOLVER_GPU_MEM_UTIL}"
  actor_rollout_ref.rollout.n=1
  actor_rollout_ref.rollout.max_num_seqs="${ROLLOUT_MAX_NUM_SEQS}"
  actor_rollout_ref.rollout.max_num_batched_tokens="${SOLVER_MAX_BATCHED_TOKENS}"
  actor_rollout_ref.rollout.response_length="${SOLVER_MAX_RESP_LEN}"
  actor_rollout_ref.rollout.enforce_eager="${ROLLOUT_ENFORCE_EAGER}"
  actor_rollout_ref.rollout.free_cache_engine="${ROLLOUT_FREE_CACHE_ENGINE}"
  actor_rollout_ref.rollout.load_format=safetensors
  actor_rollout_ref.rollout.agent.num_workers="${SOLVER_AGENT_NUM_WORKERS}"
  actor_rollout_ref.ref.fsdp_config.param_offload=True
  abstraction_actor_rollout_ref.model.path="${ABSTRACTION_MODEL_PATH}"
  abstraction_actor_rollout_ref.model.lora_rank=32
  abstraction_actor_rollout_ref.model.lora_alpha=32
  abstraction_actor_rollout_ref.model.target_modules=all-linear
  abstraction_actor_rollout_ref.model.use_remove_padding=True
  abstraction_actor_rollout_ref.model.enable_gradient_checkpointing=True
  abstraction_actor_rollout_ref.model.use_liger=True
  abstraction_actor_rollout_ref.actor.optim.lr="${ACTOR_LR}"
  abstraction_actor_rollout_ref.actor.ppo_mini_batch_size="${ACTOR_PPO_MINI_BATCH_SIZE}"
  abstraction_actor_rollout_ref.actor.use_dynamic_bsz=True
  abstraction_actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU}"
  abstraction_actor_rollout_ref.actor.use_kl_loss="${USE_KL_LOSS}"
  abstraction_actor_rollout_ref.actor.kl_loss_coef="${ACTOR_KL_LOSS_COEF}"
  abstraction_actor_rollout_ref.actor.kl_loss_type="${ACTOR_KL_LOSS_TYPE}"
  abstraction_actor_rollout_ref.actor.ulysses_sequence_parallel_size=1
  abstraction_actor_rollout_ref.actor.fsdp_config.param_offload="${ACTOR_PARAM_OFFLOAD}"
  +abstraction_actor_rollout_ref.actor.fsdp_config.grad_offload=False
  abstraction_actor_rollout_ref.actor.fsdp_config.optimizer_offload="${ACTOR_OPTIMIZER_OFFLOAD}"
  abstraction_actor_rollout_ref.rollout.tensor_model_parallel_size=1
  abstraction_actor_rollout_ref.rollout.name=vllm
  abstraction_actor_rollout_ref.rollout.dtype=bfloat16
  abstraction_actor_rollout_ref.rollout.temperature="${ABSTRACTION_ROLLOUT_TEMPERATURE}"
  abstraction_actor_rollout_ref.rollout.top_p="${ABSTRACTION_ROLLOUT_TOP_P}"
  abstraction_actor_rollout_ref.rollout.top_k="${ABSTRACTION_ROLLOUT_TOP_K}"
  abstraction_actor_rollout_ref.rollout.gpu_memory_utilization="${ABSTRACTION_GPU_MEM_UTIL}"
  abstraction_actor_rollout_ref.rollout.n=1
  abstraction_actor_rollout_ref.rollout.max_num_seqs="${ROLLOUT_MAX_NUM_SEQS}"
  abstraction_actor_rollout_ref.rollout.max_num_batched_tokens="${ABSTRACTION_MAX_BATCHED_TOKENS}"
  abstraction_actor_rollout_ref.rollout.response_length="${ABSTRACTION_MAX_RESP_LEN}"
  abstraction_actor_rollout_ref.rollout.enforce_eager="${ROLLOUT_ENFORCE_EAGER}"
  abstraction_actor_rollout_ref.rollout.free_cache_engine="${ROLLOUT_FREE_CACHE_ENGINE}"
  abstraction_actor_rollout_ref.rollout.load_format=safetensors
  abstraction_actor_rollout_ref.rollout.agent.num_workers="${ABSTRACTION_AGENT_NUM_WORKERS}"
  abstraction_actor_rollout_ref.ref.fsdp_config.param_offload=True
  two_policy.abstraction_prompt_template_path="${ABSTRACTION_PROMPT_TEMPLATE_PATH}"
  two_policy.solver_prompt_template_path="${SOLVER_PROMPT_TEMPLATE_PATH}"
  two_policy.num_abstractions="${TWO_POLICY_NUM_ABSTRACTIONS}"
  two_policy.num_solver_rollouts="${TWO_POLICY_NUM_SOLVER_ROLLOUTS}"
  two_policy.validation_num_abstractions="${TWO_POLICY_VALIDATION_NUM_ABSTRACTIONS}"
  two_policy.validation_num_solver_rollouts="${TWO_POLICY_VALIDATION_NUM_SOLVER_ROLLOUTS}"
  two_policy.validation_solver_response_length="${VALIDATION_SOLVER_MAX_RESP_LEN}"
  two_policy.decoupled_solver_schedule.enable="${TWO_POLICY_DECOUPLED_SOLVER_SCHEDULE_ENABLE}"
  two_policy.decoupled_solver_schedule.solver_update_every_n_steps="${TWO_POLICY_SOLVER_UPDATE_EVERY_N_STEPS}"
  two_policy.decoupled_solver_schedule.non_update_solver_rollouts="${TWO_POLICY_NON_UPDATE_SOLVER_ROLLOUTS}"
  two_policy.decoupled_solver_schedule.non_update_solver_temperature="${TWO_POLICY_NON_UPDATE_SOLVER_TEMPERATURE}"
  two_policy.decoupled_solver_schedule.non_update_solver_top_p="${TWO_POLICY_NON_UPDATE_SOLVER_TOP_P}"
  two_policy.decoupled_solver_schedule.non_update_solver_top_k="${TWO_POLICY_NON_UPDATE_SOLVER_TOP_K}"
  two_policy.filter_solver_groups="${TWO_POLICY_FILTER_SOLVER_GROUPS}"
  two_policy.vm_reward.enable="${TWO_POLICY_VM_REWARD_ENABLE}"
  two_policy.vm_reward.transform="${TWO_POLICY_VM_REWARD_TRANSFORM}"
  two_policy.vm_reward.solver_weight="${TWO_POLICY_VM_REWARD_SOLVER_WEIGHT}"
  two_policy.vm_reward.abstraction_weight="${TWO_POLICY_VM_REWARD_ABSTRACTION_WEIGHT}"
  trainer.critic_warmup=0
  trainer.logger=[console,wandb]
  trainer.project_name="${TRAINER_PROJECT_NAME}"
  trainer.experiment_name="${TRAINER_EXPERIMENT_NAME}"
  trainer.n_gpus_per_node=1
  trainer.nnodes=1
  trainer.rollout_data_dir="${ROLLOUT_DIR}"
  trainer.validation_data_dir="${VALIDATION_DIR}"
  trainer.default_local_dir="${CHECKPOINT_DIR}"
  trainer.save_freq="${TRAINER_SAVE_FREQ}"
  trainer.test_freq="${TRAINER_TEST_FREQ}"
  trainer.total_epochs="${TRAINER_TOTAL_EPOCHS}"
  trainer.resume_mode="${TRAINER_RESUME_MODE}"
  trainer.val_before_train="${TRAINER_VAL_BEFORE_TRAIN}"
  trainer.max_actor_ckpt_to_keep="${TRAINER_MAX_ACTOR_CKPT_TO_KEEP}"
  reward.custom_reward_function.path="${REPO_ROOT}/verl/verl/utils/reward_score/deepscaler_math_reward_multibox_patched.py"
  reward.custom_reward_function.name=compute_score
  reward.reward_manager.name=dapo
  reward.reward_kwargs.max_resp_len="${SOLVER_MAX_RESP_LEN}"
  reward.num_workers="${REWARD_NUM_WORKERS}"
  algorithm.filter_groups.enable="${FILTER_GROUPS_ENABLE}"
  algorithm.filter_groups.metric="${FILTER_GROUPS_METRIC}"
  algorithm.filter_groups.max_num_gen_batches="${FILTER_GROUPS_MAX_NUM_GEN_BATCHES}"
  ray_kwargs.ray_init.num_cpus="${RAY_NUM_CPUS}"
)

if [[ -n "${RAY_TIMELINE_JSON_FILE}" ]]; then
  args+=("ray_kwargs.timeline_json_file=${RAY_TIMELINE_JSON_FILE}")
fi

if [[ -n "${GLOBAL_PROFILER_TOOL}" ]]; then
  mkdir -p "${GLOBAL_PROFILER_SAVE_PATH}"
  args+=(
    "global_profiler.tool=${GLOBAL_PROFILER_TOOL}"
    "global_profiler.steps=${GLOBAL_PROFILER_STEPS}"
    "global_profiler.profile_continuous_steps=${GLOBAL_PROFILER_CONTINUOUS_STEPS}"
    "global_profiler.save_path=${GLOBAL_PROFILER_SAVE_PATH}"
    "actor_rollout_ref.actor.profiler.enable=${ACTOR_PROFILER_ENABLE}"
    "actor_rollout_ref.actor.profiler.all_ranks=${ACTOR_PROFILER_ALL_RANKS}"
    "actor_rollout_ref.actor.profiler.ranks=${ACTOR_PROFILER_RANKS}"
    "abstraction_actor_rollout_ref.actor.profiler.enable=${ABSTRACTION_ACTOR_PROFILER_ENABLE}"
    "abstraction_actor_rollout_ref.actor.profiler.all_ranks=${ABSTRACTION_ACTOR_PROFILER_ALL_RANKS}"
    "abstraction_actor_rollout_ref.actor.profiler.ranks=${ABSTRACTION_ACTOR_PROFILER_RANKS}"
  )
  case "${GLOBAL_PROFILER_TOOL}" in
    torch)
      args+=(
        "actor_rollout_ref.actor.profiler.tool_config.torch.contents=${ACTOR_TORCH_PROFILER_CONTENTS}"
        "abstraction_actor_rollout_ref.actor.profiler.tool_config.torch.contents=${ABSTRACTION_TORCH_PROFILER_CONTENTS}"
      )
      ;;
    torch_memory)
      args+=(
        "global_profiler.global_tool_config.torch_memory.trace_alloc_max_entries=${TORCH_MEMORY_TRACE_ALLOC_MAX_ENTRIES}"
        "global_profiler.global_tool_config.torch_memory.stack_depth=${TORCH_MEMORY_STACK_DEPTH}"
      )
      ;;
    nsys|npu)
      ;;
    *)
      echo "Unsupported GLOBAL_PROFILER_TOOL: ${GLOBAL_PROFILER_TOOL}" >&2
      exit 1
      ;;
  esac
fi

python3 -m recipe.two_policy.main_two_policy_dapo "${args[@]}" "$@"
