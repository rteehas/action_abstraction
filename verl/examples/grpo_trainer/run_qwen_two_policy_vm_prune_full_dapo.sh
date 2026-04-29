#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}

cd "${REPO_ROOT}/verl"

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1

ABSTRACTION_MODEL_PATH=${ABSTRACTION_MODEL_PATH:-${REPO_ROOT}/merged_models/qwen3_1_7b_principle_generator_ckpt1736}
SOLVER_MODEL_PATH=${SOLVER_MODEL_PATH:-Qwen/Qwen3-1.7B}
TRAIN_FILES=${TRAIN_FILES:-${REPO_ROOT}/verl_data/two_policy_deepscaler_qwne1_7b_passrate_025_075/train.parquet}
VAL_FILES=${VAL_FILES:-${REPO_ROOT}/verl_data/two_policy_aime2025_amc2023_eval/val.parquet}
ABSTRACTION_PROMPT_TEMPLATE_PATH=${ABSTRACTION_PROMPT_TEMPLATE_PATH:-${REPO_ROOT}/prompt_templates/sft_principle_generation.txt}
SOLVER_PROMPT_TEMPLATE_PATH=${SOLVER_PROMPT_TEMPLATE_PATH:-${REPO_ROOT}/prompt_templates/hint_conditioned_problem_solving_rich_v1.txt}
GPU_TUNING_PRESET=${GPU_TUNING_PRESET:-default}

case "${GPU_TUNING_PRESET}" in
  default)
    DEFAULT_SOLVER_GPU_MEM_UTIL=0.45
    DEFAULT_ABSTRACTION_GPU_MEM_UTIL=0.25
    DEFAULT_ROLLOUT_MAX_NUM_SEQS=128
    DEFAULT_SOLVER_MAX_BATCHED_TOKENS=8192
    DEFAULT_ABSTRACTION_MAX_BATCHED_TOKENS=2048
    ;;
  single_gpu_safe)
    DEFAULT_SOLVER_GPU_MEM_UTIL=0.30
    DEFAULT_ABSTRACTION_GPU_MEM_UTIL=0.16
    DEFAULT_ROLLOUT_MAX_NUM_SEQS=32
    DEFAULT_SOLVER_MAX_BATCHED_TOKENS=4096
    DEFAULT_ABSTRACTION_MAX_BATCHED_TOKENS=1024
    ;;
  h200_balanced)
    DEFAULT_SOLVER_GPU_MEM_UTIL=0.50
    DEFAULT_ABSTRACTION_GPU_MEM_UTIL=0.28
    DEFAULT_ROLLOUT_MAX_NUM_SEQS=144
    DEFAULT_SOLVER_MAX_BATCHED_TOKENS=10240
    DEFAULT_ABSTRACTION_MAX_BATCHED_TOKENS=2560
    ;;
  h200_aggressive)
    DEFAULT_SOLVER_GPU_MEM_UTIL=0.55
    DEFAULT_ABSTRACTION_GPU_MEM_UTIL=0.30
    DEFAULT_ROLLOUT_MAX_NUM_SEQS=160
    DEFAULT_SOLVER_MAX_BATCHED_TOKENS=12288
    DEFAULT_ABSTRACTION_MAX_BATCHED_TOKENS=3072
    ;;
  *)
    echo "Unknown GPU_TUNING_PRESET: ${GPU_TUNING_PRESET}" >&2
    exit 1
    ;;
esac

ABSTRACTION_MAX_RESP_LEN=${ABSTRACTION_MAX_RESP_LEN:-2048}
SOLVER_MAX_RESP_LEN=${SOLVER_MAX_RESP_LEN:-8192}
VALIDATION_SOLVER_MAX_RESP_LEN=${VALIDATION_SOLVER_MAX_RESP_LEN:-32768}
SOLVER_GPU_MEM_UTIL=${SOLVER_GPU_MEM_UTIL:-${DEFAULT_SOLVER_GPU_MEM_UTIL}}
ABSTRACTION_GPU_MEM_UTIL=${ABSTRACTION_GPU_MEM_UTIL:-${DEFAULT_ABSTRACTION_GPU_MEM_UTIL}}
ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-${DEFAULT_ROLLOUT_MAX_NUM_SEQS}}
SOLVER_MAX_BATCHED_TOKENS=${SOLVER_MAX_BATCHED_TOKENS:-${DEFAULT_SOLVER_MAX_BATCHED_TOKENS}}
ABSTRACTION_MAX_BATCHED_TOKENS=${ABSTRACTION_MAX_BATCHED_TOKENS:-${DEFAULT_ABSTRACTION_MAX_BATCHED_TOKENS}}
SOLVER_UPDATE_WEIGHTS_BUCKET_MB=${SOLVER_UPDATE_WEIGHTS_BUCKET_MB:-512}
ABSTRACTION_UPDATE_WEIGHTS_BUCKET_MB=${ABSTRACTION_UPDATE_WEIGHTS_BUCKET_MB:-512}
ACTOR_PARAM_OFFLOAD=${ACTOR_PARAM_OFFLOAD:-True}
ACTOR_OPTIMIZER_OFFLOAD=${ACTOR_OPTIMIZER_OFFLOAD:-True}
ROLLOUT_ENFORCE_EAGER=${ROLLOUT_ENFORCE_EAGER:-True}
ROLLOUT_FREE_CACHE_ENGINE=${ROLLOUT_FREE_CACHE_ENGINE:-True}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-32}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-70}
ACTOR_PPO_MINI_BATCH_SIZE=${ACTOR_PPO_MINI_BATCH_SIZE:-32}
ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU=${ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU:-32768}
TRAINER_SAVE_FREQ=${TRAINER_SAVE_FREQ:-10}
TRAINER_TEST_FREQ=${TRAINER_TEST_FREQ:-10}
TRAINER_TOTAL_EPOCHS=${TRAINER_TOTAL_EPOCHS:-10}
TRAINER_VAL_BEFORE_TRAIN=${TRAINER_VAL_BEFORE_TRAIN:-True}
TRAINER_PROJECT_NAME=${TRAINER_PROJECT_NAME:-verl_two_policy_vm_prune_dapo}
TRAINER_EXPERIMENT_NAME=${TRAINER_EXPERIMENT_NAME:-qwen3_1.7b_vm_prune}
RAY_NUM_CPUS=${RAY_NUM_CPUS:-180}
RAY_TIMELINE_JSON_FILE=${RAY_TIMELINE_JSON_FILE:-}
SOLVER_MASTER_PORT_RANGE=${SOLVER_MASTER_PORT_RANGE:-[29500,29549]}
ABSTRACTION_MASTER_PORT_RANGE=${ABSTRACTION_MASTER_PORT_RANGE:-[29550,29599]}
TWO_POLICY_DECOUPLED_SOLVER_SCHEDULE_ENABLE=${TWO_POLICY_DECOUPLED_SOLVER_SCHEDULE_ENABLE:-False}
TWO_POLICY_SOLVER_UPDATE_EVERY_N_STEPS=${TWO_POLICY_SOLVER_UPDATE_EVERY_N_STEPS:-1}
TWO_POLICY_NON_UPDATE_SOLVER_ROLLOUTS=${TWO_POLICY_NON_UPDATE_SOLVER_ROLLOUTS:-1}
TWO_POLICY_NON_UPDATE_SOLVER_TEMPERATURE=${TWO_POLICY_NON_UPDATE_SOLVER_TEMPERATURE:-0.0}
TWO_POLICY_NON_UPDATE_SOLVER_TOP_P=${TWO_POLICY_NON_UPDATE_SOLVER_TOP_P:-1.0}
TWO_POLICY_NON_UPDATE_SOLVER_TOP_K=${TWO_POLICY_NON_UPDATE_SOLVER_TOP_K:--1}
FILTER_GROUPS_ENABLE=${FILTER_GROUPS_ENABLE:-True}
FILTER_GROUPS_METRIC=${FILTER_GROUPS_METRIC:-acc}
FILTER_GROUPS_MAX_NUM_GEN_BATCHES=${FILTER_GROUPS_MAX_NUM_GEN_BATCHES:-4}
VM_PRUNING_ENABLE=${VM_PRUNING_ENABLE:-True}
VM_PRUNING_APPLY_IN_VALIDATION=${VM_PRUNING_APPLY_IN_VALIDATION:-False}
VM_MODEL_PATH=${VM_MODEL_PATH:-VGS-AI/DeepSeek-VM-1.5B}
VM_TOKENIZER_PATH=${VM_TOKENIZER_PATH:-deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B}
VM_ATTN_IMPLEMENTATION=${VM_ATTN_IMPLEMENTATION:-flash_attention_2}
VM_TORCH_DTYPE=${VM_TORCH_DTYPE:-bfloat16}
VM_DEVICE=${VM_DEVICE:-}
VM_DEVICE_MAP=${VM_DEVICE_MAP:-auto}
VM_SCORING_BATCH_SIZE=${VM_SCORING_BATCH_SIZE:-8}
VM_BLOCK_LENGTH=${VM_BLOCK_LENGTH:-4096}
VM_SOLVER_KEEP_FRACTION=${VM_SOLVER_KEEP_FRACTION:-0.5}
VM_SOLVER_MIN_KEEP=${VM_SOLVER_MIN_KEEP:-1}
VM_SOLVER_KEEP_COUNT=${VM_SOLVER_KEEP_COUNT:-}
VM_SOLVER_ROUNDING=${VM_SOLVER_ROUNDING:-ceil}
VM_ABSTRACTION_KEEP_FRACTION=${VM_ABSTRACTION_KEEP_FRACTION:-0.5}
VM_ABSTRACTION_MIN_KEEP=${VM_ABSTRACTION_MIN_KEEP:-1}
VM_ABSTRACTION_KEEP_COUNT=${VM_ABSTRACTION_KEEP_COUNT:-}
VM_ABSTRACTION_ROUNDING=${VM_ABSTRACTION_ROUNDING:-ceil}
MAX_RESP_LEN=$(( ABSTRACTION_MAX_RESP_LEN > SOLVER_MAX_RESP_LEN ? ABSTRACTION_MAX_RESP_LEN : SOLVER_MAX_RESP_LEN ))
RUN_TAG=${RUN_TAG:-$(date -u +%Y%m%d_%H%M%S)}
RUN_ROOT=${RUN_ROOT:-/tmp/action_abstraction/two_policy_runs}
RUN_DIR=${RUN_DIR:-${RUN_ROOT}/full_vm_prune_dapo_${RUN_TAG}}
ROLLOUT_DIR=${ROLLOUT_DIR:-${RUN_DIR}/rollouts}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-${RUN_DIR}/checkpoints}
VALIDATION_DIR=${VALIDATION_DIR:-${RUN_DIR}/validation_rollouts}
VM_OFFLOAD_ENABLE=${VM_OFFLOAD_ENABLE:-False}
VM_OFFLOAD_FOLDER=${VM_OFFLOAD_FOLDER:-${RUN_DIR}/vm_offload}
VM_OFFLOAD_STATE_DICT=${VM_OFFLOAD_STATE_DICT:-True}
VM_LOW_CPU_MEM_USAGE=${VM_LOW_CPU_MEM_USAGE:-True}
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

mkdir -p "${RUN_DIR}" "${ROLLOUT_DIR}" "${CHECKPOINT_DIR}" "${VALIDATION_DIR}"
if [[ "${VM_OFFLOAD_ENABLE}" = "True" ]]; then
  mkdir -p "${VM_OFFLOAD_FOLDER}"
fi

args=(
  hydra.searchpath=[file://${REPO_ROOT}/verl/verl/trainer/config]
  algorithm.adv_estimator=grpo
  data.train_files="${TRAIN_FILES}"
  data.val_files="${VAL_FILES}"
  data.train_batch_size="${TRAIN_BATCH_SIZE}"
  data.val_batch_size="${VAL_BATCH_SIZE}"
  data.max_prompt_length=4096
  data.max_response_length="${MAX_RESP_LEN}"
  actor_rollout_ref.model.path="${SOLVER_MODEL_PATH}"
  actor_rollout_ref.model.lora_rank=32
  actor_rollout_ref.model.lora_alpha=32
  actor_rollout_ref.model.target_modules=all-linear
  actor_rollout_ref.model.use_remove_padding=True
  actor_rollout_ref.model.enable_gradient_checkpointing=True
  actor_rollout_ref.model.use_liger=True
  actor_rollout_ref.actor.optim.lr=1e-6
  actor_rollout_ref.actor.ppo_mini_batch_size="${ACTOR_PPO_MINI_BATCH_SIZE}"
  actor_rollout_ref.actor.use_dynamic_bsz=True
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU}"
  actor_rollout_ref.actor.use_kl_loss=True
  actor_rollout_ref.actor.kl_loss_coef=0.001
  actor_rollout_ref.actor.kl_loss_type=low_var_kl
  actor_rollout_ref.actor.ulysses_sequence_parallel_size=1
  actor_rollout_ref.actor.fsdp_config.param_offload="${ACTOR_PARAM_OFFLOAD}"
  +actor_rollout_ref.actor.fsdp_config.grad_offload=False
  actor_rollout_ref.actor.fsdp_config.optimizer_offload="${ACTOR_OPTIMIZER_OFFLOAD}"
  actor_rollout_ref.rollout.tensor_model_parallel_size=1
  actor_rollout_ref.rollout.name=vllm
  actor_rollout_ref.rollout.dtype=bfloat16
  actor_rollout_ref.rollout.temperature=0.6
  actor_rollout_ref.rollout.top_p=0.95
  actor_rollout_ref.rollout.top_k=20
  actor_rollout_ref.rollout.gpu_memory_utilization="${SOLVER_GPU_MEM_UTIL}"
  actor_rollout_ref.rollout.n=1
  actor_rollout_ref.rollout.max_num_seqs="${ROLLOUT_MAX_NUM_SEQS}"
  actor_rollout_ref.rollout.max_num_batched_tokens="${SOLVER_MAX_BATCHED_TOKENS}"
  actor_rollout_ref.rollout.response_length="${SOLVER_MAX_RESP_LEN}"
  actor_rollout_ref.rollout.enforce_eager="${ROLLOUT_ENFORCE_EAGER}"
  actor_rollout_ref.rollout.free_cache_engine="${ROLLOUT_FREE_CACHE_ENGINE}"
  actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes="${SOLVER_UPDATE_WEIGHTS_BUCKET_MB}"
  actor_rollout_ref.rollout.load_format=safetensors
  actor_rollout_ref.ref.fsdp_config.param_offload=True
  abstraction_actor_rollout_ref.model.path="${ABSTRACTION_MODEL_PATH}"
  abstraction_actor_rollout_ref.model.lora_rank=32
  abstraction_actor_rollout_ref.model.lora_alpha=32
  abstraction_actor_rollout_ref.model.target_modules=all-linear
  abstraction_actor_rollout_ref.model.use_remove_padding=True
  abstraction_actor_rollout_ref.model.enable_gradient_checkpointing=True
  abstraction_actor_rollout_ref.model.use_liger=True
  abstraction_actor_rollout_ref.actor.optim.lr=1e-6
  abstraction_actor_rollout_ref.actor.ppo_mini_batch_size="${ACTOR_PPO_MINI_BATCH_SIZE}"
  abstraction_actor_rollout_ref.actor.use_dynamic_bsz=True
  abstraction_actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU}"
  abstraction_actor_rollout_ref.actor.use_kl_loss=True
  abstraction_actor_rollout_ref.actor.kl_loss_coef=0.001
  abstraction_actor_rollout_ref.actor.kl_loss_type=low_var_kl
  abstraction_actor_rollout_ref.actor.ulysses_sequence_parallel_size=1
  abstraction_actor_rollout_ref.actor.fsdp_config.param_offload="${ACTOR_PARAM_OFFLOAD}"
  +abstraction_actor_rollout_ref.actor.fsdp_config.grad_offload=False
  abstraction_actor_rollout_ref.actor.fsdp_config.optimizer_offload="${ACTOR_OPTIMIZER_OFFLOAD}"
  abstraction_actor_rollout_ref.rollout.tensor_model_parallel_size=1
  abstraction_actor_rollout_ref.rollout.name=vllm
  abstraction_actor_rollout_ref.rollout.dtype=bfloat16
  abstraction_actor_rollout_ref.rollout.temperature=0.6
  abstraction_actor_rollout_ref.rollout.top_p=0.95
  abstraction_actor_rollout_ref.rollout.top_k=20
  abstraction_actor_rollout_ref.rollout.gpu_memory_utilization="${ABSTRACTION_GPU_MEM_UTIL}"
  abstraction_actor_rollout_ref.rollout.n=1
  abstraction_actor_rollout_ref.rollout.max_num_seqs="${ROLLOUT_MAX_NUM_SEQS}"
  abstraction_actor_rollout_ref.rollout.max_num_batched_tokens="${ABSTRACTION_MAX_BATCHED_TOKENS}"
  abstraction_actor_rollout_ref.rollout.response_length="${ABSTRACTION_MAX_RESP_LEN}"
  abstraction_actor_rollout_ref.rollout.enforce_eager="${ROLLOUT_ENFORCE_EAGER}"
  abstraction_actor_rollout_ref.rollout.free_cache_engine="${ROLLOUT_FREE_CACHE_ENGINE}"
  abstraction_actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes="${ABSTRACTION_UPDATE_WEIGHTS_BUCKET_MB}"
  abstraction_actor_rollout_ref.rollout.load_format=safetensors
  abstraction_actor_rollout_ref.ref.fsdp_config.param_offload=True
  two_policy.abstraction_prompt_template_path="${ABSTRACTION_PROMPT_TEMPLATE_PATH}"
  two_policy.solver_prompt_template_path="${SOLVER_PROMPT_TEMPLATE_PATH}"
  two_policy.num_abstractions=4
  two_policy.num_solver_rollouts=8
  two_policy.validation_num_abstractions=4
  two_policy.validation_num_solver_rollouts=4
  two_policy.validation_solver_response_length="${VALIDATION_SOLVER_MAX_RESP_LEN}"
  two_policy.decoupled_solver_schedule.enable="${TWO_POLICY_DECOUPLED_SOLVER_SCHEDULE_ENABLE}"
  two_policy.decoupled_solver_schedule.solver_update_every_n_steps="${TWO_POLICY_SOLVER_UPDATE_EVERY_N_STEPS}"
  two_policy.decoupled_solver_schedule.non_update_solver_rollouts="${TWO_POLICY_NON_UPDATE_SOLVER_ROLLOUTS}"
  two_policy.decoupled_solver_schedule.non_update_solver_temperature="${TWO_POLICY_NON_UPDATE_SOLVER_TEMPERATURE}"
  two_policy.decoupled_solver_schedule.non_update_solver_top_p="${TWO_POLICY_NON_UPDATE_SOLVER_TOP_P}"
  two_policy.decoupled_solver_schedule.non_update_solver_top_k="${TWO_POLICY_NON_UPDATE_SOLVER_TOP_K}"
  two_policy.vm_pruning.enable="${VM_PRUNING_ENABLE}"
  two_policy.vm_pruning.apply_in_validation="${VM_PRUNING_APPLY_IN_VALIDATION}"
  two_policy.vm_pruning.model_path="${VM_MODEL_PATH}"
  two_policy.vm_pruning.tokenizer_path="${VM_TOKENIZER_PATH}"
  two_policy.vm_pruning.attn_implementation="${VM_ATTN_IMPLEMENTATION}"
  two_policy.vm_pruning.torch_dtype="${VM_TORCH_DTYPE}"
  two_policy.vm_pruning.scoring_batch_size="${VM_SCORING_BATCH_SIZE}"
  two_policy.vm_pruning.block_length="${VM_BLOCK_LENGTH}"
  two_policy.vm_pruning.offload.enable="${VM_OFFLOAD_ENABLE}"
  two_policy.vm_pruning.offload.folder="${VM_OFFLOAD_FOLDER}"
  two_policy.vm_pruning.offload.state_dict="${VM_OFFLOAD_STATE_DICT}"
  two_policy.vm_pruning.offload.low_cpu_mem_usage="${VM_LOW_CPU_MEM_USAGE}"
  two_policy.vm_pruning.solver_update.keep_fraction="${VM_SOLVER_KEEP_FRACTION}"
  two_policy.vm_pruning.solver_update.min_keep="${VM_SOLVER_MIN_KEEP}"
  two_policy.vm_pruning.solver_update.rounding="${VM_SOLVER_ROUNDING}"
  two_policy.vm_pruning.abstraction_update.keep_fraction="${VM_ABSTRACTION_KEEP_FRACTION}"
  two_policy.vm_pruning.abstraction_update.min_keep="${VM_ABSTRACTION_MIN_KEEP}"
  two_policy.vm_pruning.abstraction_update.rounding="${VM_ABSTRACTION_ROUNDING}"
  trainer.critic_warmup=0
  trainer.logger=[console,wandb]
  trainer.project_name="${TRAINER_PROJECT_NAME}"
  trainer.experiment_name="${TRAINER_EXPERIMENT_NAME}"
  trainer.n_gpus_per_node=1
  trainer.nnodes=1
  trainer.actor_master_port_range="${SOLVER_MASTER_PORT_RANGE}"
  trainer.abstraction_master_port_range="${ABSTRACTION_MASTER_PORT_RANGE}"
  trainer.rollout_data_dir="${ROLLOUT_DIR}"
  trainer.validation_data_dir="${VALIDATION_DIR}"
  trainer.default_local_dir="${CHECKPOINT_DIR}"
  trainer.save_freq="${TRAINER_SAVE_FREQ}"
  trainer.test_freq="${TRAINER_TEST_FREQ}"
  trainer.total_epochs="${TRAINER_TOTAL_EPOCHS}"
  trainer.resume_mode=auto
  trainer.val_before_train="${TRAINER_VAL_BEFORE_TRAIN}"
  trainer.max_actor_ckpt_to_keep=1
  reward.custom_reward_function.path="${REPO_ROOT}/verl/verl/utils/reward_score/deepscaler_math_reward_multibox_patched.py"
  reward.custom_reward_function.name=compute_score
  algorithm.filter_groups.enable="${FILTER_GROUPS_ENABLE}"
  algorithm.filter_groups.metric="${FILTER_GROUPS_METRIC}"
  algorithm.filter_groups.max_num_gen_batches="${FILTER_GROUPS_MAX_NUM_GEN_BATCHES}"
  reward.reward_manager.name=dapo
  reward.reward_kwargs.max_resp_len="${SOLVER_MAX_RESP_LEN}"
  ray_kwargs.ray_init.num_cpus="${RAY_NUM_CPUS}"
)

if [[ -n "${VM_DEVICE}" ]]; then
  args+=("two_policy.vm_pruning.device=${VM_DEVICE}")
  args+=("two_policy.vm_pruning.device_map=none")
elif [[ -n "${VM_DEVICE_MAP}" ]]; then
  args+=("two_policy.vm_pruning.device_map=${VM_DEVICE_MAP}")
fi
if [[ -n "${VM_SOLVER_KEEP_COUNT}" ]]; then
  args+=("two_policy.vm_pruning.solver_update.keep_count=${VM_SOLVER_KEEP_COUNT}")
fi
if [[ -n "${VM_ABSTRACTION_KEEP_COUNT}" ]]; then
  args+=("two_policy.vm_pruning.abstraction_update.keep_count=${VM_ABSTRACTION_KEEP_COUNT}")
fi
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

python3 -m recipe.two_policy.main_two_policy_vm_prune_dapo "${args[@]}" "$@"
