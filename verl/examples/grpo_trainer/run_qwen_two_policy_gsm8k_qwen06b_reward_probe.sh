#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# Reward-probe defaults: skip pre-train validation, keep the training batch
# small, and push eval/save far enough out that they should not trigger during
# a short exploratory run.
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-8}
export TRAINER_VAL_BEFORE_TRAIN=${TRAINER_VAL_BEFORE_TRAIN:-False}
export TRAINER_SAVE_FREQ=${TRAINER_SAVE_FREQ:-999999999}
export TRAINER_TEST_FREQ=${TRAINER_TEST_FREQ:-999999999}
export TRAINER_EXPERIMENT_NAME=${TRAINER_EXPERIMENT_NAME:-qwen06b_gsm8k_two_policy_reward_probe}
export RUN_TAG=${RUN_TAG:-reward_probe_$(date -u +%Y%m%d_%H%M%S)}

exec "${SCRIPT_DIR}/run_qwen_two_policy_gsm8k_qwen06b_explicit.sh" "$@"
