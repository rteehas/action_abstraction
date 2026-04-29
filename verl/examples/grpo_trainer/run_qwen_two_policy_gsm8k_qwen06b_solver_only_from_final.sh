#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}

ORIGINAL_RUN_DIR=${ORIGINAL_RUN_DIR:-/scratch/rst306/action_abstractions/action_abstraction/verl_checkpoints/gsm8k_two_policy_nodecoupled_solverlr5e7_abslr1e7_b64_v80_abs4_sol4_ppomb_sweep_ckpt315/h200/2026-04-24/slurm_7049337_task_0_pmb32}
ORIGINAL_CHECKPOINT_DIR=${ORIGINAL_CHECKPOINT_DIR:-${ORIGINAL_RUN_DIR}/checkpoints}
ORIGINAL_TOTAL_EPOCHS=${ORIGINAL_TOTAL_EPOCHS:-30}
RESUME_STEP=${RESUME_STEP:-520}
RESUME_FROM_PATH=${RESUME_FROM_PATH:-${ORIGINAL_CHECKPOINT_DIR}/global_step_${RESUME_STEP}}
ADDITIONAL_EPOCHS=${ADDITIONAL_EPOCHS:-10}
TRAINER_TOTAL_EPOCHS=${TRAINER_TOTAL_EPOCHS:-$((ORIGINAL_TOTAL_EPOCHS + ADDITIONAL_EPOCHS))}

if [[ ! -d "${RESUME_FROM_PATH}" ]]; then
    echo "Resume checkpoint not found: ${RESUME_FROM_PATH}" >&2
    exit 1
fi

if (( TRAINER_TOTAL_EPOCHS <= ORIGINAL_TOTAL_EPOCHS )); then
    echo "TRAINER_TOTAL_EPOCHS (${TRAINER_TOTAL_EPOCHS}) must exceed ORIGINAL_TOTAL_EPOCHS (${ORIGINAL_TOTAL_EPOCHS}) to continue training from the final checkpoint." >&2
    exit 1
fi

RUN_ROOT=${RUN_ROOT:-${ORIGINAL_RUN_DIR}/solver_only_continuations}
TRAINER_PROJECT_NAME=${TRAINER_PROJECT_NAME:-verl_two_policy_gsm8k_qwen06b}
TRAINER_EXPERIMENT_NAME=${TRAINER_EXPERIMENT_NAME:-qwen06b_gsm8k_two_policy_solver_only_from_final}
RUN_TAG=${RUN_TAG:-solver_only_from_gs${RESUME_STEP}_$(date -u +%Y%m%d_%H%M%S)}
RUN_DIR=${RUN_DIR:-${RUN_ROOT}/${RUN_TAG}}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-${RUN_DIR}/checkpoints}
ROLLOUT_DIR=${ROLLOUT_DIR:-${RUN_DIR}/rollouts}
VALIDATION_DIR=${VALIDATION_DIR:-${RUN_DIR}/validation_rollouts}
WANDB_DIR=${WANDB_DIR:-${RUN_DIR}/wandb}

mkdir -p "${RUN_DIR}" "${CHECKPOINT_DIR}" "${ROLLOUT_DIR}" "${VALIDATION_DIR}" "${WANDB_DIR}"

export REPO_ROOT
export TRAINER_PROJECT_NAME
export TRAINER_EXPERIMENT_NAME
export RUN_ROOT
export RUN_TAG
export RUN_DIR
export CHECKPOINT_DIR
export ROLLOUT_DIR
export VALIDATION_DIR
export WANDB_DIR

export DATASET_DIR=${DATASET_DIR:-${REPO_ROOT}/verl_data/gsm8k_two_policy_passrate_le_0625}
export TRAIN_FILES=${TRAIN_FILES:-${DATASET_DIR}/train.parquet}
export VAL_FILES=${VAL_FILES:-${DATASET_DIR}/val.parquet}
export ABSTRACTION_MODEL_PATH=${ABSTRACTION_MODEL_PATH:-${REPO_ROOT}/merged_models/qwen3_0_6b_gsm8k_insight_abstraction_generation_lora32_ep10_ckpt315}
export SOLVER_MODEL_PATH=${SOLVER_MODEL_PATH:-Qwen/Qwen3-0.6B}
export ABSTRACTION_PROMPT_TEMPLATE_PATH=${ABSTRACTION_PROMPT_TEMPLATE_PATH:-${REPO_ROOT}/prompt_templates/insight_abstraction_generation_sft_template.txt}
export SOLVER_PROMPT_TEMPLATE_PATH=${SOLVER_PROMPT_TEMPLATE_PATH:-${REPO_ROOT}/prompt_templates/hint_conditioned_problem_solving_rich_v1.txt}
export REWARD_FN_PATH=${REWARD_FN_PATH:-${REPO_ROOT}/verl/verl/utils/reward_score/deepscaler_math_reward_multibox_patched.py}

export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}
export VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-80}
export MAX_PROMPT_LEN=${MAX_PROMPT_LEN:-2048}
export ABSTRACTION_MAX_RESP_LEN=${ABSTRACTION_MAX_RESP_LEN:-512}
export SOLVER_MAX_RESP_LEN=${SOLVER_MAX_RESP_LEN:-1024}
export VALIDATION_SOLVER_MAX_RESP_LEN=${VALIDATION_SOLVER_MAX_RESP_LEN:-1024}
export VALIDATION_NUM_ABSTRACTIONS=${VALIDATION_NUM_ABSTRACTIONS:-1}
export VALIDATION_NUM_SOLVER_ROLLOUTS=${VALIDATION_NUM_SOLVER_ROLLOUTS:-1}
export ACTOR_PPO_MINI_BATCH_SIZE=${ACTOR_PPO_MINI_BATCH_SIZE:-32}
export ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU=${ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU:-16384}
export SOLVER_GPU_MEM_UTIL=${SOLVER_GPU_MEM_UTIL:-0.5}
export ABSTRACTION_GPU_MEM_UTIL=${ABSTRACTION_GPU_MEM_UTIL:-0.3}
export TRAINER_SAVE_FREQ=${TRAINER_SAVE_FREQ:-20}
export TRAINER_TEST_FREQ=${TRAINER_TEST_FREQ:-100}
export TRAINER_TOTAL_EPOCHS
export TRAINER_VAL_BEFORE_TRAIN=${TRAINER_VAL_BEFORE_TRAIN:-False}
export RAY_NUM_CPUS=${RAY_NUM_CPUS:-12}

# Solver-only continuation: solver updates every step; abstraction keeps
# generating the same number of rollouts with the same sampling settings, but
# never enters PPO preparation/update.
export DECOUPLED_SOLVER_SCHEDULE_ENABLE=${DECOUPLED_SOLVER_SCHEDULE_ENABLE:-True}
export SOLVER_SCHEDULE_MODE=${SOLVER_SCHEDULE_MODE:-periodic}
export SOLVER_UPDATE_EVERY_N_STEPS=${SOLVER_UPDATE_EVERY_N_STEPS:-1}
export SOLVER_UPDATE_BLOCK_SIZE=${SOLVER_UPDATE_BLOCK_SIZE:-1}
export SOLVER_SCHEDULE_START_WITH=${SOLVER_SCHEDULE_START_WITH:-solver}
export NON_UPDATE_SOLVER_ROLLOUTS=${NON_UPDATE_SOLVER_ROLLOUTS:-1}
export NON_UPDATE_SOLVER_TEMPERATURE=${NON_UPDATE_SOLVER_TEMPERATURE:-0.0}
export NON_UPDATE_SOLVER_TOP_P=${NON_UPDATE_SOLVER_TOP_P:-1.0}
export NON_UPDATE_SOLVER_TOP_K=${NON_UPDATE_SOLVER_TOP_K:--1}
export DECOUPLED_ABSTRACTION_SCHEDULE_ENABLE=${DECOUPLED_ABSTRACTION_SCHEDULE_ENABLE:-True}
export ABSTRACTION_USE_SOLVER_STEP_COMPLEMENT=${ABSTRACTION_USE_SOLVER_STEP_COMPLEMENT:-True}
export NON_UPDATE_ABSTRACTIONS=${NON_UPDATE_ABSTRACTIONS:-4}
export NON_UPDATE_ABSTRACTION_TEMPERATURE=${NON_UPDATE_ABSTRACTION_TEMPERATURE:-0.6}
export NON_UPDATE_ABSTRACTION_TOP_P=${NON_UPDATE_ABSTRACTION_TOP_P:-1.0}
export NON_UPDATE_ABSTRACTION_TOP_K=${NON_UPDATE_ABSTRACTION_TOP_K:--1}
export TWO_POLICY_FILTER_SOLVER_GROUPS=${TWO_POLICY_FILTER_SOLVER_GROUPS:-False}
export TWO_POLICY_REQUIRE_PRINCIPLE_HEADERS=${TWO_POLICY_REQUIRE_PRINCIPLE_HEADERS:-False}

echo "Solver-only two-policy continuation:"
echo "  original_run_dir=${ORIGINAL_RUN_DIR}"
echo "  resume_from_path=${RESUME_FROM_PATH}"
echo "  original_total_epochs=${ORIGINAL_TOTAL_EPOCHS}"
echo "  trainer_total_epochs=${TRAINER_TOTAL_EPOCHS}"
echo "  run_dir=${RUN_DIR}"
echo "  checkpoint_dir=${CHECKPOINT_DIR}"
echo "  rollout_dir=${ROLLOUT_DIR}"
echo "  validation_dir=${VALIDATION_DIR}"
echo "  abstraction_updates=disabled_via_schedule"
echo "  solver_updates=every_step"

args=(
    actor_rollout_ref.actor.optim.lr=5e-7
    abstraction_actor_rollout_ref.actor.optim.lr=1e-7
    actor_rollout_ref.rollout.top_p=1
    actor_rollout_ref.rollout.top_k=-1
    actor_rollout_ref.rollout.max_num_batched_tokens=8192
    actor_rollout_ref.rollout.max_num_seqs=1024
    abstraction_actor_rollout_ref.rollout.top_p=1
    abstraction_actor_rollout_ref.rollout.top_k=-1
    abstraction_actor_rollout_ref.rollout.max_num_batched_tokens=8192
    abstraction_actor_rollout_ref.rollout.max_num_seqs=1024
    actor_rollout_ref.rollout.agent.num_workers=8
    abstraction_actor_rollout_ref.rollout.agent.num_workers=8
    data.dataloader_num_workers=8
    two_policy.num_abstractions=4
    two_policy.num_solver_rollouts=4
    trainer.max_actor_ckpt_to_keep=2
    trainer.resume_mode=resume_path
    trainer.resume_from_path="${RESUME_FROM_PATH}"
)

exec "${SCRIPT_DIR}/run_qwen_two_policy_gsm8k_qwen06b_explicit.sh" "${args[@]}" "$@"
