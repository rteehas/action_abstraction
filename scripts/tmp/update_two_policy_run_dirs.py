from pathlib import Path

updates = {
    Path("/workspace/action_abstraction/verl/examples/grpo_trainer/run_qwen_two_policy_full.sh"): [
        (
            """ABSTRACTION_MODEL_PATH=${ABSTRACTION_MODEL_PATH:-/workspace/action_abstraction/merged_models/qwen3_1_7b_principle_generator_ckpt1736}\nSOLVER_MODEL_PATH=${SOLVER_MODEL_PATH:-Qwen/Qwen3-1.7B}\nTRAIN_FILES=${TRAIN_FILES:-/workspace/action_abstraction/verl_data/two_policy_deepscaler_qwne1_7b_passrate_025_075/train.parquet}\nVAL_FILES=${VAL_FILES:-/workspace/action_abstraction/verl_data/two_policy_aime2025_amc2023_eval/val.parquet}\nABSTRACTION_MAX_RESP_LEN=${ABSTRACTION_MAX_RESP_LEN:-1024}\nSOLVER_MAX_RESP_LEN=${SOLVER_MAX_RESP_LEN:-8192}\nMAX_RESP_LEN=$(( ABSTRACTION_MAX_RESP_LEN > SOLVER_MAX_RESP_LEN ? ABSTRACTION_MAX_RESP_LEN : SOLVER_MAX_RESP_LEN ))\n""",
            """ABSTRACTION_MODEL_PATH=${ABSTRACTION_MODEL_PATH:-/workspace/action_abstraction/merged_models/qwen3_1_7b_principle_generator_ckpt1736}\nSOLVER_MODEL_PATH=${SOLVER_MODEL_PATH:-Qwen/Qwen3-1.7B}\nTRAIN_FILES=${TRAIN_FILES:-/workspace/action_abstraction/verl_data/two_policy_deepscaler_qwne1_7b_passrate_025_075/train.parquet}\nVAL_FILES=${VAL_FILES:-/workspace/action_abstraction/verl_data/two_policy_aime2025_amc2023_eval/val.parquet}\nABSTRACTION_MAX_RESP_LEN=${ABSTRACTION_MAX_RESP_LEN:-1024}\nSOLVER_MAX_RESP_LEN=${SOLVER_MAX_RESP_LEN:-8192}\nMAX_RESP_LEN=$(( ABSTRACTION_MAX_RESP_LEN > SOLVER_MAX_RESP_LEN ? ABSTRACTION_MAX_RESP_LEN : SOLVER_MAX_RESP_LEN ))\nRUN_TAG=${RUN_TAG:-$(date -u +%Y%m%d_%H%M%S)}\nRUN_ROOT=${RUN_ROOT:-/tmp/action_abstraction/two_policy_runs}\nRUN_DIR=${RUN_DIR:-${RUN_ROOT}/full_grpo_${RUN_TAG}}\nROLLOUT_DIR=${ROLLOUT_DIR:-${RUN_DIR}/rollouts}\nCHECKPOINT_DIR=${CHECKPOINT_DIR:-${RUN_DIR}/checkpoints}\n\nmkdir -p "${RUN_DIR}" "${ROLLOUT_DIR}" "${CHECKPOINT_DIR}"\n""",
        ),
        (
            """  trainer.n_gpus_per_node=1\n  trainer.nnodes=1\n  trainer.rollout_data_dir=""\n  trainer.default_local_dir=""\n""",
            """  trainer.n_gpus_per_node=1\n  trainer.nnodes=1\n  trainer.rollout_data_dir="${ROLLOUT_DIR}"\n  trainer.default_local_dir="${CHECKPOINT_DIR}"\n""",
        ),
    ],
    Path("/workspace/action_abstraction/verl/examples/grpo_trainer/run_qwen_two_policy_full_dapo.sh"): [
        (
            """ABSTRACTION_MODEL_PATH=${ABSTRACTION_MODEL_PATH:-/workspace/action_abstraction/merged_models/qwen3_1_7b_principle_generator_ckpt1736}\nSOLVER_MODEL_PATH=${SOLVER_MODEL_PATH:-Qwen/Qwen3-1.7B}\nTRAIN_FILES=${TRAIN_FILES:-/workspace/action_abstraction/verl_data/two_policy_deepscaler_qwne1_7b_passrate_025_075/train.parquet}\nVAL_FILES=${VAL_FILES:-/workspace/action_abstraction/verl_data/two_policy_aime2025_amc2023_eval/val.parquet}\nABSTRACTION_MAX_RESP_LEN=${ABSTRACTION_MAX_RESP_LEN:-1024}\nSOLVER_MAX_RESP_LEN=${SOLVER_MAX_RESP_LEN:-8192}\nMAX_RESP_LEN=$(( ABSTRACTION_MAX_RESP_LEN > SOLVER_MAX_RESP_LEN ? ABSTRACTION_MAX_RESP_LEN : SOLVER_MAX_RESP_LEN ))\n""",
            """ABSTRACTION_MODEL_PATH=${ABSTRACTION_MODEL_PATH:-/workspace/action_abstraction/merged_models/qwen3_1_7b_principle_generator_ckpt1736}\nSOLVER_MODEL_PATH=${SOLVER_MODEL_PATH:-Qwen/Qwen3-1.7B}\nTRAIN_FILES=${TRAIN_FILES:-/workspace/action_abstraction/verl_data/two_policy_deepscaler_qwne1_7b_passrate_025_075/train.parquet}\nVAL_FILES=${VAL_FILES:-/workspace/action_abstraction/verl_data/two_policy_aime2025_amc2023_eval/val.parquet}\nABSTRACTION_MAX_RESP_LEN=${ABSTRACTION_MAX_RESP_LEN:-1024}\nSOLVER_MAX_RESP_LEN=${SOLVER_MAX_RESP_LEN:-8192}\nMAX_RESP_LEN=$(( ABSTRACTION_MAX_RESP_LEN > SOLVER_MAX_RESP_LEN ? ABSTRACTION_MAX_RESP_LEN : SOLVER_MAX_RESP_LEN ))\nRUN_TAG=${RUN_TAG:-$(date -u +%Y%m%d_%H%M%S)}\nRUN_ROOT=${RUN_ROOT:-/tmp/action_abstraction/two_policy_runs}\nRUN_DIR=${RUN_DIR:-${RUN_ROOT}/full_dapo_${RUN_TAG}}\nROLLOUT_DIR=${ROLLOUT_DIR:-${RUN_DIR}/rollouts}\nCHECKPOINT_DIR=${CHECKPOINT_DIR:-${RUN_DIR}/checkpoints}\n\nmkdir -p "${RUN_DIR}" "${ROLLOUT_DIR}" "${CHECKPOINT_DIR}"\n""",
        ),
        (
            """  trainer.n_gpus_per_node=1\n  trainer.nnodes=1\n  trainer.rollout_data_dir=""\n  trainer.default_local_dir=""\n""",
            """  trainer.n_gpus_per_node=1\n  trainer.nnodes=1\n  trainer.rollout_data_dir="${ROLLOUT_DIR}"\n  trainer.default_local_dir="${CHECKPOINT_DIR}"\n""",
        ),
    ],
    Path("/workspace/action_abstraction/verl/examples/grpo_trainer/run_qwen_two_policy_tiny_smoke.sh"): [
        (
            """RUN_DIR=${RUN_DIR:-/workspace/action_abstraction/outputs/2026-03-24/two_policy_smoke_${MODE}_${RUN_TAG}}\n\nmkdir -p "${RUN_DIR}"\n""",
            """RUN_ROOT=${RUN_ROOT:-/tmp/action_abstraction/two_policy_runs}\nRUN_DIR=${RUN_DIR:-${RUN_ROOT}/tiny_smoke_${MODE}_${RUN_TAG}}\nROLLOUT_DIR=${ROLLOUT_DIR:-${RUN_DIR}/rollouts}\nCHECKPOINT_DIR=${CHECKPOINT_DIR:-${RUN_DIR}/checkpoints}\n\nmkdir -p "${RUN_DIR}" "${ROLLOUT_DIR}" "${CHECKPOINT_DIR}"\n""",
        ),
        (
            """  trainer.rollout_data_dir="${RUN_DIR}/rollouts"\n  trainer.default_local_dir="${RUN_DIR}/checkpoints"\n""",
            """  trainer.rollout_data_dir="${ROLLOUT_DIR}"\n  trainer.default_local_dir="${CHECKPOINT_DIR}"\n""",
        ),
    ],
    Path("/workspace/action_abstraction/verl/examples/grpo_trainer/run_qwen_two_policy_tiny_overfit.sh"): [
        (
            """RUN_DIR=${RUN_DIR:-/workspace/action_abstraction/outputs/2026-03-24/two_policy_overfit_${MODE}_${RUN_TAG}}\nEPOCHS=${EPOCHS:-20}\nLR=${LR:-1e-6}\nNUM_ABS=${NUM_ABS:-2}\nNUM_SOL=${NUM_SOL:-4}\n\nmkdir -p "${RUN_DIR}"\n""",
            """RUN_ROOT=${RUN_ROOT:-/tmp/action_abstraction/two_policy_runs}\nRUN_DIR=${RUN_DIR:-${RUN_ROOT}/tiny_overfit_${MODE}_${RUN_TAG}}\nROLLOUT_DIR=${ROLLOUT_DIR:-${RUN_DIR}/rollouts}\nCHECKPOINT_DIR=${CHECKPOINT_DIR:-${RUN_DIR}/checkpoints}\nEPOCHS=${EPOCHS:-20}\nLR=${LR:-1e-6}\nNUM_ABS=${NUM_ABS:-2}\nNUM_SOL=${NUM_SOL:-4}\n\nmkdir -p "${RUN_DIR}" "${ROLLOUT_DIR}" "${CHECKPOINT_DIR}"\n""",
        ),
        (
            """  trainer.rollout_data_dir="${RUN_DIR}/rollouts"\n  trainer.default_local_dir="${RUN_DIR}/checkpoints"\n""",
            """  trainer.rollout_data_dir="${ROLLOUT_DIR}"\n  trainer.default_local_dir="${CHECKPOINT_DIR}"\n""",
        ),
        (
            """LATEST_CKPT=$(find "${RUN_DIR}/checkpoints" -maxdepth 1 -type d -name "global_step_*" | sort -V | tail -n 1)\nif [ -z "${LATEST_CKPT}" ]; then\n  echo "No checkpoint found under ${RUN_DIR}/checkpoints" >&2\n""",
            """LATEST_CKPT=$(find "${CHECKPOINT_DIR}" -maxdepth 1 -type d -name "global_step_*" | sort -V | tail -n 1)\nif [ -z "${LATEST_CKPT}" ]; then\n  echo "No checkpoint found under ${CHECKPOINT_DIR}" >&2\n""",
        ),
    ],
}

for path, replacements in updates.items():
    text = path.read_text()
    original = text
    for old, new in replacements:
        if old not in text:
            raise SystemExit(f"pattern not found in {path}:\n{old}")
        text = text.replace(old, new, 1)
    if text != original:
        path.write_text(text)
