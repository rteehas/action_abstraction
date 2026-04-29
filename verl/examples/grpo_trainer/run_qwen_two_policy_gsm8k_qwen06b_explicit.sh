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

DATASET_DIR=${DATASET_DIR:-${REPO_ROOT}/verl_data/gsm8k_two_policy_passrate_le_0625}
TRAIN_FILES=${TRAIN_FILES:-${DATASET_DIR}/train.parquet}
VAL_FILES=${VAL_FILES:-${DATASET_DIR}/val.parquet}

ABSTRACTION_ADAPTER_PATH=${ABSTRACTION_ADAPTER_PATH:-${REPO_ROOT}/sft_models/Qwen3_0_6B-gsm8k_insight_abstraction_generation_qwen3_0p6b/20260420_210815/checkpoint-252}
ABSTRACTION_MODEL_PATH=${ABSTRACTION_MODEL_PATH:-${REPO_ROOT}/merged_models/qwen3_0_6b_gsm8k_insight_abstraction_generation_ckpt252}
SOLVER_MODEL_PATH=${SOLVER_MODEL_PATH:-Qwen/Qwen3-0.6B}

ABSTRACTION_PROMPT_TEMPLATE_PATH=${ABSTRACTION_PROMPT_TEMPLATE_PATH:-${REPO_ROOT}/prompt_templates/insight_abstraction_generation_sft_template.txt}
SOLVER_PROMPT_TEMPLATE_PATH=${SOLVER_PROMPT_TEMPLATE_PATH:-${REPO_ROOT}/prompt_templates/hint_conditioned_problem_solving_rich_v1.txt}
REWARD_FN_PATH=${REWARD_FN_PATH:-${REPO_ROOT}/verl/verl/utils/reward_score/deepscaler_math_reward_multibox_patched.py}

TRAINER_PROJECT_NAME=${TRAINER_PROJECT_NAME:-verl_two_policy_gsm8k_qwen06b}
TRAINER_EXPERIMENT_NAME=${TRAINER_EXPERIMENT_NAME:-qwen06b_gsm8k_two_policy}
RUN_ROOT=${RUN_ROOT:-${REPO_ROOT}/verl_checkpoints/two_policy_gsm8k_qwen06b}
RUN_TAG=${RUN_TAG:-qwen06b_gsm8k_two_policy_$(date -u +%Y%m%d_%H%M%S)}
RUN_DIR=${RUN_DIR:-${RUN_ROOT}/${TRAINER_EXPERIMENT_NAME}_${RUN_TAG}}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-${RUN_DIR}/checkpoints}
ROLLOUT_DIR=${ROLLOUT_DIR:-${RUN_DIR}/rollouts}
VALIDATION_DIR=${VALIDATION_DIR:-${RUN_DIR}/validation_rollouts}
WANDB_DIR=${WANDB_DIR:-${RUN_DIR}/wandb}
TRAINER_LOGGER=${TRAINER_LOGGER:-'["console","wandb"]'}
TRAINER_ROLLOUT_DATA_DIR=${TRAINER_ROLLOUT_DATA_DIR:-${ROLLOUT_DIR}}
TRAINER_VALIDATION_DATA_DIR=${TRAINER_VALIDATION_DATA_DIR:-${VALIDATION_DIR}}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-8}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-8}
MAX_PROMPT_LEN=${MAX_PROMPT_LEN:-2048}
ABSTRACTION_MAX_RESP_LEN=${ABSTRACTION_MAX_RESP_LEN:-512}
SOLVER_MAX_RESP_LEN=${SOLVER_MAX_RESP_LEN:-1024}
VALIDATION_SOLVER_MAX_RESP_LEN=${VALIDATION_SOLVER_MAX_RESP_LEN:-1024}
VALIDATION_NUM_ABSTRACTIONS=${VALIDATION_NUM_ABSTRACTIONS:-5}
VALIDATION_NUM_SOLVER_ROLLOUTS=${VALIDATION_NUM_SOLVER_ROLLOUTS:-5}
MAX_RESP_LEN=$(( ABSTRACTION_MAX_RESP_LEN > SOLVER_MAX_RESP_LEN ? ABSTRACTION_MAX_RESP_LEN : SOLVER_MAX_RESP_LEN ))
ACTOR_PPO_MINI_BATCH_SIZE=${ACTOR_PPO_MINI_BATCH_SIZE:-8}
ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU=${ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU:-16384}
SOLVER_GPU_MEM_UTIL=${SOLVER_GPU_MEM_UTIL:-0.45}
ABSTRACTION_GPU_MEM_UTIL=${ABSTRACTION_GPU_MEM_UTIL:-0.25}
TRAINER_SAVE_FREQ=${TRAINER_SAVE_FREQ:-100}
TRAINER_TEST_FREQ=${TRAINER_TEST_FREQ:-100}
TRAINER_TOTAL_EPOCHS=${TRAINER_TOTAL_EPOCHS:-15}
TRAINER_VAL_BEFORE_TRAIN=${TRAINER_VAL_BEFORE_TRAIN:-True}
RAY_NUM_CPUS=${RAY_NUM_CPUS:-8}
DECOUPLED_SOLVER_SCHEDULE_ENABLE=${DECOUPLED_SOLVER_SCHEDULE_ENABLE:-False}
SOLVER_SCHEDULE_MODE=${SOLVER_SCHEDULE_MODE:-periodic}
SOLVER_UPDATE_EVERY_N_STEPS=${SOLVER_UPDATE_EVERY_N_STEPS:-2}
SOLVER_UPDATE_BLOCK_SIZE=${SOLVER_UPDATE_BLOCK_SIZE:-1}
SOLVER_SCHEDULE_START_WITH=${SOLVER_SCHEDULE_START_WITH:-solver}
NON_UPDATE_SOLVER_ROLLOUTS=${NON_UPDATE_SOLVER_ROLLOUTS:-1}
NON_UPDATE_SOLVER_TEMPERATURE=${NON_UPDATE_SOLVER_TEMPERATURE:-0.0}
NON_UPDATE_SOLVER_TOP_P=${NON_UPDATE_SOLVER_TOP_P:-1.0}
NON_UPDATE_SOLVER_TOP_K=${NON_UPDATE_SOLVER_TOP_K:--1}
DECOUPLED_ABSTRACTION_SCHEDULE_ENABLE=${DECOUPLED_ABSTRACTION_SCHEDULE_ENABLE:-False}
ABSTRACTION_USE_SOLVER_STEP_COMPLEMENT=${ABSTRACTION_USE_SOLVER_STEP_COMPLEMENT:-True}
NON_UPDATE_ABSTRACTIONS=${NON_UPDATE_ABSTRACTIONS:-1}
NON_UPDATE_ABSTRACTION_TEMPERATURE=${NON_UPDATE_ABSTRACTION_TEMPERATURE:-0.0}
NON_UPDATE_ABSTRACTION_TOP_P=${NON_UPDATE_ABSTRACTION_TOP_P:-1.0}
NON_UPDATE_ABSTRACTION_TOP_K=${NON_UPDATE_ABSTRACTION_TOP_K:--1}
TWO_POLICY_FILTER_SOLVER_GROUPS=${TWO_POLICY_FILTER_SOLVER_GROUPS:-True}
TWO_POLICY_REQUIRE_PRINCIPLE_HEADERS=${TWO_POLICY_REQUIRE_PRINCIPLE_HEADERS:-True}

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
    if [[ ! -f "${ABSTRACTION_ADAPTER_PATH}/adapter_config.json" ]]; then
        echo "Abstraction model not found and adapter checkpoint missing: ${ABSTRACTION_ADAPTER_PATH}" >&2
        exit 1
    fi

    mkdir -p "${ABSTRACTION_MODEL_PATH}"
    python3 "${REPO_ROOT}/scripts/merge_lora_adapter.py" \
        --base-model Qwen/Qwen3-0.6B \
        --lora-path "${ABSTRACTION_ADAPTER_PATH}" \
        --output-dir "${ABSTRACTION_MODEL_PATH}"
fi

if [[ ! -f "${ABSTRACTION_MODEL_PATH}/config.json" ]]; then
    echo "Merged abstraction model is incomplete: ${ABSTRACTION_MODEL_PATH}" >&2
    exit 1
fi

mkdir -p "${RUN_DIR}" "${CHECKPOINT_DIR}" "${ROLLOUT_DIR}" "${VALIDATION_DIR}" "${WANDB_DIR}"

export WANDB_DIR

echo "Two-policy GSM8K Qwen 0.6B launch:"
echo "  train_files=${TRAIN_FILES}"
echo "  val_files=${VAL_FILES}"
echo "  abstraction_adapter=${ABSTRACTION_ADAPTER_PATH}"
echo "  abstraction_model=${ABSTRACTION_MODEL_PATH}"
echo "  solver_model=${SOLVER_MODEL_PATH}"
echo "  abstraction_prompt_template=${ABSTRACTION_PROMPT_TEMPLATE_PATH}"
echo "  solver_prompt_template=${SOLVER_PROMPT_TEMPLATE_PATH}"
echo "  run_root=${RUN_ROOT}"
echo "  run_tag=${RUN_TAG}"
echo "  run_dir=${RUN_DIR}"
echo "  checkpoint_dir=${CHECKPOINT_DIR}"
echo "  rollout_dir=${ROLLOUT_DIR}"
echo "  validation_dir=${VALIDATION_DIR}"
echo "  wandb_dir=${WANDB_DIR}"
echo "  decoupled_solver_schedule=${DECOUPLED_SOLVER_SCHEDULE_ENABLE}"
echo "  solver_schedule_mode=${SOLVER_SCHEDULE_MODE}"
echo "  solver_update_every_n_steps=${SOLVER_UPDATE_EVERY_N_STEPS}"
echo "  solver_update_block_size=${SOLVER_UPDATE_BLOCK_SIZE}"
echo "  solver_schedule_start_with=${SOLVER_SCHEDULE_START_WITH}"
echo "  validation_num_abstractions=${VALIDATION_NUM_ABSTRACTIONS}"
echo "  validation_num_solver_rollouts=${VALIDATION_NUM_SOLVER_ROLLOUTS}"
echo "  decoupled_abstraction_schedule=${DECOUPLED_ABSTRACTION_SCHEDULE_ENABLE}"
echo "  abstraction_use_solver_step_complement=${ABSTRACTION_USE_SOLVER_STEP_COMPLEMENT}"
echo "  filter_solver_groups=${TWO_POLICY_FILTER_SOLVER_GROUPS}"
echo "  require_principle_headers=${TWO_POLICY_REQUIRE_PRINCIPLE_HEADERS}"

python3 -m recipe.two_policy.main_two_policy_ppo \
    hydra.searchpath=[file://${REPO_ROOT}/verl/verl/trainer/config] \
    algorithm.adv_estimator=grpo \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${VAL_FILES}" \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.val_batch_size="${VAL_BATCH_SIZE}" \
    data.max_prompt_length="${MAX_PROMPT_LEN}" \
    data.max_response_length="${MAX_RESP_LEN}" \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    actor_rollout_ref.model.path="${SOLVER_MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_liger=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.ppo_mini_batch_size="${ACTOR_PPO_MINI_BATCH_SIZE}" \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU}" \
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
    actor_rollout_ref.rollout.gpu_memory_utilization="${SOLVER_GPU_MEM_UTIL}" \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.response_length="${SOLVER_MAX_RESP_LEN}" \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    abstraction_actor_rollout_ref.model.path="${ABSTRACTION_MODEL_PATH}" \
    abstraction_actor_rollout_ref.model.use_remove_padding=True \
    abstraction_actor_rollout_ref.model.enable_gradient_checkpointing=True \
    abstraction_actor_rollout_ref.model.use_liger=True \
    abstraction_actor_rollout_ref.actor.optim.lr=1e-6 \
    abstraction_actor_rollout_ref.actor.entropy_coeff=0 \
    abstraction_actor_rollout_ref.actor.ppo_mini_batch_size="${ACTOR_PPO_MINI_BATCH_SIZE}" \
    abstraction_actor_rollout_ref.actor.use_dynamic_bsz=True \
    abstraction_actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU}" \
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
    abstraction_actor_rollout_ref.rollout.gpu_memory_utilization="${ABSTRACTION_GPU_MEM_UTIL}" \
    abstraction_actor_rollout_ref.rollout.n=1 \
    abstraction_actor_rollout_ref.rollout.response_length="${ABSTRACTION_MAX_RESP_LEN}" \
    abstraction_actor_rollout_ref.rollout.enforce_eager=True \
    abstraction_actor_rollout_ref.rollout.free_cache_engine=True \
    abstraction_actor_rollout_ref.rollout.load_format=safetensors \
    abstraction_actor_rollout_ref.ref.fsdp_config.param_offload=True \
    two_policy.abstraction_prompt_template_path="${ABSTRACTION_PROMPT_TEMPLATE_PATH}" \
    two_policy.solver_prompt_template_path="${SOLVER_PROMPT_TEMPLATE_PATH}" \
    two_policy.num_abstractions=5 \
    two_policy.num_solver_rollouts=5 \
    two_policy.validation_num_abstractions="${VALIDATION_NUM_ABSTRACTIONS}" \
    two_policy.validation_num_solver_rollouts="${VALIDATION_NUM_SOLVER_ROLLOUTS}" \
    two_policy.validation_solver_response_length="${VALIDATION_SOLVER_MAX_RESP_LEN}" \
    two_policy.require_principle_headers="${TWO_POLICY_REQUIRE_PRINCIPLE_HEADERS}" \
    two_policy.filter_solver_groups="${TWO_POLICY_FILTER_SOLVER_GROUPS}" \
    two_policy.decoupled_solver_schedule.enable=${DECOUPLED_SOLVER_SCHEDULE_ENABLE} \
    two_policy.decoupled_solver_schedule.mode="${SOLVER_SCHEDULE_MODE}" \
    two_policy.decoupled_solver_schedule.solver_update_every_n_steps="${SOLVER_UPDATE_EVERY_N_STEPS}" \
    two_policy.decoupled_solver_schedule.block_size="${SOLVER_UPDATE_BLOCK_SIZE}" \
    two_policy.decoupled_solver_schedule.start_with="${SOLVER_SCHEDULE_START_WITH}" \
    two_policy.decoupled_solver_schedule.non_update_solver_rollouts="${NON_UPDATE_SOLVER_ROLLOUTS}" \
    two_policy.decoupled_solver_schedule.non_update_solver_temperature="${NON_UPDATE_SOLVER_TEMPERATURE}" \
    two_policy.decoupled_solver_schedule.non_update_solver_top_p="${NON_UPDATE_SOLVER_TOP_P}" \
    two_policy.decoupled_solver_schedule.non_update_solver_top_k="${NON_UPDATE_SOLVER_TOP_K}" \
    two_policy.decoupled_abstraction_schedule.enable=${DECOUPLED_ABSTRACTION_SCHEDULE_ENABLE} \
    two_policy.decoupled_abstraction_schedule.use_solver_update_complement=${ABSTRACTION_USE_SOLVER_STEP_COMPLEMENT} \
    two_policy.decoupled_abstraction_schedule.non_update_abstractions="${NON_UPDATE_ABSTRACTIONS}" \
    two_policy.decoupled_abstraction_schedule.non_update_abstraction_temperature="${NON_UPDATE_ABSTRACTION_TEMPERATURE}" \
    two_policy.decoupled_abstraction_schedule.non_update_abstraction_top_p="${NON_UPDATE_ABSTRACTION_TOP_P}" \
    two_policy.decoupled_abstraction_schedule.non_update_abstraction_top_k="${NON_UPDATE_ABSTRACTION_TOP_K}" \
    two_policy.vm_reward.enable=False \
    two_policy.vm_reward.transform=logit \
    two_policy.vm_reward.solver_weight=0.1 \
    two_policy.vm_reward.abstraction_weight=0.1 \
    trainer.critic_warmup=0 \
    trainer.logger="${TRAINER_LOGGER}" \
    trainer.project_name="${TRAINER_PROJECT_NAME}" \
    trainer.experiment_name="${TRAINER_EXPERIMENT_NAME}" \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.rollout_data_dir="${TRAINER_ROLLOUT_DATA_DIR}" \
    trainer.validation_data_dir="${TRAINER_VALIDATION_DATA_DIR}" \
    trainer.default_local_dir="${CHECKPOINT_DIR}" \
    trainer.save_freq="${TRAINER_SAVE_FREQ}" \
    trainer.test_freq="${TRAINER_TEST_FREQ}" \
    trainer.total_epochs="${TRAINER_TOTAL_EPOCHS}" \
    trainer.resume_mode=auto \
    trainer.val_before_train="${TRAINER_VAL_BEFORE_TRAIN}" \
    trainer.max_actor_ckpt_to_keep=1 \
    reward.custom_reward_function.path="${REWARD_FN_PATH}" \
    reward.custom_reward_function.name=compute_score \
    reward.reward_manager.name=naive \
    ray_kwargs.ray_init.num_cpus="${RAY_NUM_CPUS}" \
    "$@"
