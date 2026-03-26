#!/bin/bash
set -euo pipefail

RUN_PID=${RUN_PID:-492613}
ROOT=${ROOT:-/workspace/action_abstraction/outputs/2026-03-22/contrastive_abstraction_prompting/rlad_top4_rlad_solver_template_sweep}
LOG=${LOG:-$ROOT/finalize.log}

source /workspace/miniconda3/etc/profile.d/conda.sh
conda activate abstraction

echo "[watcher] waiting for top4 rerun pid $RUN_PID" >> "$LOG"
while kill -0 "$RUN_PID" 2>/dev/null; do
  date -u +"%Y-%m-%dT%H:%M:%SZ [watcher] rerun still running" >> "$LOG"
  sleep 120
done

date -u +"%Y-%m-%dT%H:%M:%SZ [watcher] rerun exited" >> "$LOG"
/workspace/miniconda3/envs/abstraction/bin/python /tmp/generate_top4_rlad_solver_template_report.py --root "$ROOT" >> "$LOG" 2>&1

echo "[watcher] final report generated" >> "$LOG"
