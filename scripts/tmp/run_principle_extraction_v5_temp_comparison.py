from __future__ import annotations

import json
import sys
from pathlib import Path

from datasets import load_from_disk
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

REPO_ROOT = Path("/workspace/action_abstraction")
SCRIPTS_DIR = REPO_ROOT / "scripts"
for p in (REPO_ROOT, SCRIPTS_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from contrastive_abstraction_utils import format_solution_block, read_text, strip_think_blocks

MODEL_NAME = "Qwen/Qwen3-1.7B"
DATASET_PATH = REPO_ROOT / "contrastive_abstraction_datasets/deepscaler_passrate_gt0_3k"
PROMPT_PATH = REPO_ROOT / "prompt_templates/principle_extraction_template_v5.txt"
OUTPUT_DIR = REPO_ROOT / "outputs/2026-03-20/contrastive_abstraction_prompting/principle_extraction_v5_temp06_comparison"
ROWS_PATH = OUTPUT_DIR / "rows.json"
MARKDOWN_PATH = OUTPUT_DIR / "samples.md"

def render_prompt(template: str, problem: str, trace: str) -> str:
    return template.replace("{{PROBLEM}}", problem).replace("{{CORRECT_SOLUTIONS_BLOCK}}", trace)

def first_nonempty_correct_trace(example: dict) -> str | None:
    for trace in example["selected_correct_solutions"]:
        cleaned = strip_think_blocks(trace)
        if cleaned:
            return cleaned
    return None

def first_row_with_passrate_and_trace(ds, target: float):
    for i in range(len(ds)):
        ex = ds[i]
        if float(ex["passrate"]) != target:
            continue
        trace = first_nonempty_correct_trace(ex)
        if trace:
            return i, ex, trace
    raise RuntimeError(f"No row found with passrate={target} and a non-empty correct trace")

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ds = load_from_disk(str(DATASET_PATH))["train"]
    selected = [
        first_row_with_passrate_and_trace(ds, 1.0),
        first_row_with_passrate_and_trace(ds, 0.25),
    ]

    template = read_text(PROMPT_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    prompts = []
    rows = []
    for idx, ex, trace in selected:
        trace_block = format_solution_block("Correct Trace", [trace], max_chars=4500)
        prompt = render_prompt(template, ex["problem"], trace_block)
        prompts.append(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        )
        rows.append({
            "source_index": idx,
            "row_id": ex["row_id"],
            "passrate": ex["passrate"],
            "num_correct": ex["num_correct"],
            "answer": ex["answer"],
            "problem": ex["problem"],
            "trace_excerpt": trace[:1200],
        })

    llm = LLM(
        model=MODEL_NAME,
        enable_lora=False,
        max_model_len=16384,
        max_num_batched_tokens=4096,
        gpu_memory_utilization=0.9,
    )

    greedy_sampling = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        max_tokens=1200,
        seed=0,
    )
    greedy_outputs = llm.generate(prompts, greedy_sampling)

    temp_sampling = SamplingParams(
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        n=2,
        max_tokens=1200,
        seed=0,
    )
    temp_outputs = llm.generate(prompts, temp_sampling)

    saved = []
    for meta, greedy, temp in zip(rows, greedy_outputs, temp_outputs):
        temp_texts = [(o.text or "").strip() for o in temp.outputs]
        record = {
            **meta,
            "greedy_output": (greedy.outputs[0].text or "").strip(),
            "temp06_outputs": temp_texts,
        }
        saved.append(record)

    ROWS_PATH.write_text(json.dumps(saved, indent=2), encoding="utf-8")

    md = []
    md.append("# Principle Extraction V5 Temperature Comparison")
    md.append("")
    md.append(f"- model: {MODEL_NAME!r}")
    md[-1] = md[-1].replace("'Qwen/Qwen3-1.7B'", "`Qwen/Qwen3-1.7B`")
    md.append(f"- prompt: `{PROMPT_PATH.relative_to(REPO_ROOT)}`")
    md.append("- rows: first train row with passrate `1.0` and first train row with passrate `0.25`, each requiring a non-empty first correct trace")
    md.append("- comparison: greedy (`temperature=0.0`) vs sampled (`temperature=0.6`, `n=2`)")
    md.append("")

    for row in saved:
        md.append(f"## Row {row['row_id']} (passrate={row['passrate']}, num_correct={row['num_correct']})")
        md.append("")
        md.append("**Problem**")
        md.append(row["problem"].strip())
        md.append("")
        md.append("**Greedy (`temperature=0.0`)**")
        md.append("")
        md.append("```text")
        md.append(row["greedy_output"].rstrip())
        md.append("```")
        md.append("")
        for i, text in enumerate(row["temp06_outputs"], start=1):
            md.append(f"**Sample {i} (`temperature=0.6`)**")
            md.append("")
            md.append("```text")
            md.append(text.rstrip())
            md.append("```")
            md.append("")
        md.append("---")
        md.append("")

    MARKDOWN_PATH.write_text("\n".join(md), encoding="utf-8")
    print(json.dumps({
        "rows": len(saved),
        "output_dir": str(OUTPUT_DIR),
        "markdown": str(MARKDOWN_PATH),
    }, indent=2))

if __name__ == "__main__":
    main()
