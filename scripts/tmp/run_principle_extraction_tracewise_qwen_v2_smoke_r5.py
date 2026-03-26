from __future__ import annotations

import importlib.util
from pathlib import Path

print("wrapper:start", flush=True)
SCRIPT_PATH = Path("/tmp/run_principle_extraction_tracewise_qwen_v2.py")
SMOKE_DIR = Path("/workspace/action_abstraction/outputs/2026-03-20/contrastive_abstraction_prompting/principle_extractions_tracewise_qwen_v2_smoke_r5")
print(f"wrapper:script_exists={SCRIPT_PATH.exists()}", flush=True)

def main() -> None:
    spec = importlib.util.spec_from_file_location("tracewise_qwen_v2", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load script spec from {SCRIPT_PATH}")
    print("wrapper:before_exec_module", flush=True)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    print("wrapper:after_exec_module", flush=True)

    mod.OUTPUT_DIR = SMOKE_DIR
    mod.REPORT_PATH = SMOKE_DIR / "report.json"
    mod.ROWS_PATH = SMOKE_DIR / "rows.json"
    mod.MARKDOWN_PATH = SMOKE_DIR / "principles_by_problem.md"
    print("wrapper:before_main", flush=True)
    mod.main()
    print("wrapper:after_main", flush=True)

if __name__ == "__main__":
    main()
