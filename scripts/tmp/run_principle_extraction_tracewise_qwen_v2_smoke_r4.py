from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT_PATH = Path("/tmp/run_principle_extraction_tracewise_qwen_v2.py")
SMOKE_DIR = Path("/workspace/action_abstraction/outputs/2026-03-20/contrastive_abstraction_prompting/principle_extractions_tracewise_qwen_v2_smoke_r4")

def main() -> None:
    spec = importlib.util.spec_from_file_location("tracewise_qwen_v2", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load script spec from {SCRIPT_PATH}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    mod.OUTPUT_DIR = SMOKE_DIR
    mod.REPORT_PATH = SMOKE_DIR / "report.json"
    mod.ROWS_PATH = SMOKE_DIR / "rows.json"
    mod.MARKDOWN_PATH = SMOKE_DIR / "principles_by_problem.md"
    mod.main()

if __name__ == "__main__":
    main()
