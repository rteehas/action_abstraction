#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}

export REPO_ROOT

# Keep run artifacts on /scratch instead of inheriting the base script's /tmp default.
export RUN_ROOT=${RUN_ROOT:-/scratch/rst306/action_abstractions/action_abstraction/two_policy_runs}
export RUN_TAG=${RUN_TAG:-schedule_b_$(date -u +%Y%m%d_%H%M%S)}

# Requested response-length settings.
export ABSTRACTION_MAX_RESP_LEN=${ABSTRACTION_MAX_RESP_LEN:-2048}
export SOLVER_MAX_RESP_LEN=${SOLVER_MAX_RESP_LEN:-8192}

# Proposal 6 / Schedule B defaults.
export TWO_POLICY_DECOUPLED_SOLVER_SCHEDULE_ENABLE=${TWO_POLICY_DECOUPLED_SOLVER_SCHEDULE_ENABLE:-True}
export TWO_POLICY_SOLVER_UPDATE_EVERY_N_STEPS=${TWO_POLICY_SOLVER_UPDATE_EVERY_N_STEPS:-4}
export TWO_POLICY_NON_UPDATE_SOLVER_ROLLOUTS=${TWO_POLICY_NON_UPDATE_SOLVER_ROLLOUTS:-1}
export TWO_POLICY_NON_UPDATE_SOLVER_TEMPERATURE=${TWO_POLICY_NON_UPDATE_SOLVER_TEMPERATURE:-0.0}
export TWO_POLICY_NON_UPDATE_SOLVER_TOP_P=${TWO_POLICY_NON_UPDATE_SOLVER_TOP_P:-1.0}
export TWO_POLICY_NON_UPDATE_SOLVER_TOP_K=${TWO_POLICY_NON_UPDATE_SOLVER_TOP_K:--1}
export TWO_POLICY_VM_REWARD_ENABLE=${TWO_POLICY_VM_REWARD_ENABLE:-False}
export TWO_POLICY_VM_REWARD_TRANSFORM=${TWO_POLICY_VM_REWARD_TRANSFORM:-logit}
export TWO_POLICY_VM_REWARD_SOLVER_WEIGHT=${TWO_POLICY_VM_REWARD_SOLVER_WEIGHT:-0.1}
export TWO_POLICY_VM_REWARD_ABSTRACTION_WEIGHT=${TWO_POLICY_VM_REWARD_ABSTRACTION_WEIGHT:-0.1}

# Tiny overfit launcher uses the non-VM trainer; keep GRPO as the default mode.
export MODE=${MODE:-grpo}

exec bash "${SCRIPT_DIR}/run_qwen_two_policy_tiny_overfit.sh" "$@"
