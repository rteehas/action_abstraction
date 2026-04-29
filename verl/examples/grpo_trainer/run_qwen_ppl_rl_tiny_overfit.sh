#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}

if [[ -z "${CONDA_DEFAULT_ENV:-}" ]] && [[ -f /workspace/miniconda3/etc/profile.d/conda.sh ]]; then
  source /workspace/miniconda3/etc/profile.d/conda.sh
  conda activate abstraction
fi

RUN_TAG=${RUN_TAG:-$(date -u +%Y%m%d_%H%M%S)}
DATA_DIR=${DATA_DIR:-/tmp/action_abstraction/ppl_rl_tiny_overfit}
SOURCE_DATASET=${SOURCE_DATASET:-${REPO_ROOT}/deepscaler_32k_baseline_solutions_scored}
ACTOR_MODEL_PATH=${ACTOR_MODEL_PATH:-${REPO_ROOT}/merged_models/qwen3_1_7b_principle_generator_ckpt1736}
PPL_MODEL_PATH=${PPL_MODEL_PATH:-Qwen/Qwen3-1.7B}
ABSTRACTION_PROMPT_TEMPLATE_PATH=${ABSTRACTION_PROMPT_TEMPLATE_PATH:-${REPO_ROOT}/prompt_templates/sft_principle_generation.txt}
SOLVER_PROMPT_TEMPLATE_PATH=${SOLVER_PROMPT_TEMPLATE_PATH:-${REPO_ROOT}/prompt_templates/hint_conditioned_problem_solving_rich_v1.txt}
TRAIN_SIZE=${TRAIN_SIZE:-2}
VAL_SIZE=${VAL_SIZE:-2}
VAL_SAME_AS_TRAIN=${VAL_SAME_AS_TRAIN:-True}
RUN_ROOT=${RUN_ROOT:-/tmp/action_abstraction/ppl_rl_runs}
RUN_DIR=${RUN_DIR:-${RUN_ROOT}/tiny_overfit_${RUN_TAG}}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-${RUN_DIR}/checkpoints}
TRAINER_TOTAL_EPOCHS=${TRAINER_TOTAL_EPOCHS:-20}
LR=${LR:-5e-5}
REWARD_CLIP=${REWARD_CLIP:-False}
POSITIVE_IMPROVEMENT_ONLY=${POSITIVE_IMPROVEMENT_ONLY:-False}

mkdir -p "${RUN_DIR}" "${CHECKPOINT_DIR}"

build_args=(
  --source-dataset "${SOURCE_DATASET}"
  --output-dir "${DATA_DIR}"
  --model-name "${PPL_MODEL_PATH}"
  --abstraction-prompt-template "${ABSTRACTION_PROMPT_TEMPLATE_PATH}"
  --solver-prompt-template "${SOLVER_PROMPT_TEMPLATE_PATH}"
  --min-correct-solutions 2
  --train-size "${TRAIN_SIZE}"
  --val-size "${VAL_SIZE}"
)

if [[ "${VAL_SAME_AS_TRAIN}" == "True" ]]; then
  build_args+=(--val-same-as-train)
fi

python "${REPO_ROOT}/scripts/build_ppl_rl_dataset.py" "${build_args[@]}"

COMMON_ENV=(
  REPO_ROOT="${REPO_ROOT}"
  TRAIN_FILES="${DATA_DIR}/train.parquet"
  VAL_FILES="${DATA_DIR}/val.parquet"
  ACTOR_MODEL_PATH="${ACTOR_MODEL_PATH}"
  PPL_MODEL_PATH="${PPL_MODEL_PATH}"
  SOLVER_PROMPT_TEMPLATE_PATH="${SOLVER_PROMPT_TEMPLATE_PATH}"
  RUN_DIR="${RUN_DIR}"
  CHECKPOINT_DIR="${CHECKPOINT_DIR}"
  LR="${LR}"
  REWARD_CLIP="${REWARD_CLIP}"
  POSITIVE_IMPROVEMENT_ONLY="${POSITIVE_IMPROVEMENT_ONLY}"
  TRAINER_PROJECT_NAME="ppl_rl_overfit"
  TRAINER_EXPERIMENT_NAME="tiny_ppl_rl_${RUN_TAG}"
)

env "${COMMON_ENV[@]}" bash "${SCRIPT_DIR}/run_qwen_ppl_rl.sh" \
  trainer.val_before_train=True \
  trainer.val_only=True \
  trainer.resume_mode=disable \
  | tee "${RUN_DIR}/baseline_eval.log"

ray stop --force >/dev/null 2>&1 || true

env "${COMMON_ENV[@]}" TRAINER_TOTAL_EPOCHS="${TRAINER_TOTAL_EPOCHS}" bash "${SCRIPT_DIR}/run_qwen_ppl_rl.sh" \
  trainer.val_before_train=True \
  trainer.val_only=False \
  trainer.resume_mode=disable \
  | tee "${RUN_DIR}/train.log"

ray stop --force >/dev/null 2>&1 || true

LATEST_CKPT=$(find "${CHECKPOINT_DIR}" -maxdepth 1 -type d -name "global_step_*" | sort -V | tail -n 1)
if [[ -z "${LATEST_CKPT}" ]]; then
  echo "No checkpoint found under ${CHECKPOINT_DIR}" >&2
  exit 1
fi

env "${COMMON_ENV[@]}" bash "${SCRIPT_DIR}/run_qwen_ppl_rl.sh" \
  trainer.val_before_train=True \
  trainer.val_only=True \
  trainer.resume_mode=resume_path \
  trainer.resume_from_path="${LATEST_CKPT}" \
  | tee "${RUN_DIR}/final_eval.log"

python - <<'PY' "${RUN_DIR}/baseline_eval.log" "${RUN_DIR}/final_eval.log"
import ast
import re
import sys
from pathlib import Path


def parse_metrics(path: str):
    text = Path(path).read_text()
    matches = re.findall(r"Initial validation metrics: (\{.*?\})", text)
    if not matches:
        raise SystemExit(f"Could not find validation metrics in {path}")
    return ast.literal_eval(matches[-1])


baseline = parse_metrics(sys.argv[1])
final = parse_metrics(sys.argv[2])
interesting = sorted(
    key
    for key in set(baseline).intersection(final)
    if any(token in key.lower() for token in ["reward", "score", "ppl", "valid", "abstraction"])
)

print("Baseline metrics:")
for key in interesting:
    print(f"  {key}: {baseline[key]}")

print("Final metrics:")
for key in interesting:
    print(f"  {key}: {final[key]}")

print("Delta:")
for key in interesting:
    try:
        delta = final[key] - baseline[key]
    except TypeError:
        continue
    print(f"  {key}: {delta}")
PY
