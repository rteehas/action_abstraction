from __future__ import annotations
import json
import sys
from pathlib import Path

import torch
from datasets import load_from_disk, DatasetDict
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

REPO_ROOT = Path('/workspace/action_abstraction')
SCRIPTS_DIR = REPO_ROOT / 'scripts'
for p in (REPO_ROOT, SCRIPTS_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from contrastive_abstraction_utils import read_text, strip_think_blocks

MODEL_NAME = 'Qwen/Qwen3-1.7B'
DATASET_PATH = REPO_ROOT / 'contrastive_abstraction_datasets/deepscaler_mixed_correct_incorrect_256_all_correct'
PROMPT_PATH = REPO_ROOT / 'prompt_templates/principle_extraction_template_v3.txt'
OUTPUT_DIR = REPO_ROOT / 'outputs/2026-03-19/contrastive_abstraction_prompting/principle_extractions_tracewise_qwen_v3_r1'
REPORT_PATH = OUTPUT_DIR / 'report.json'
ROWS_PATH = OUTPUT_DIR / 'rows.json'
MARKDOWN_PATH = OUTPUT_DIR / 'principles_by_problem.md'


def render_prompt(template: str, problem: str, trace: str) -> str:
    prompt = template.replace('{{PROBLEM}}', problem)
    prompt = prompt.replace('{{CORRECT_SOLUTIONS_BLOCK}}', trace)
    return prompt


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset = load_from_disk(str(DATASET_PATH))
    if isinstance(dataset, DatasetDict):
        ds = dataset['test']
    else:
        ds = dataset
    source_rows = [dict(ds[i]) for i in range(min(26, ds.num_rows))]

    template = read_text(PROMPT_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    prompt_rows: list[dict] = []
    prompts: list[str] = []
    messages: list[str] = []

    for row in source_rows:
        cleaned_traces = [strip_think_blocks(t) for t in row['selected_correct_solutions']]
        for local_idx, trace in enumerate(cleaned_traces, start=1):
            if not trace:
                continue
            prompt = render_prompt(template, row['problem'], trace)
            prompt_rows.append(
                {
                    'row_id': row['row_id'],
                    'problem': row['problem'],
                    'answer': row['answer'],
                    'trace_local_idx': local_idx,
                    'trace_text': trace,
                }
            )
            prompts.append(prompt)
            messages.append(
                tokenizer.apply_chat_template(
                    [{'role': 'user', 'content': prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            )

    print(f'Device name = {torch.cuda.get_device_name(0)}')
    llm = LLM(
        model=MODEL_NAME,
        enable_lora=False,
        max_model_len=16384,
        max_num_batched_tokens=4096,
        gpu_memory_utilization=0.9,
    )
    sampling = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        max_tokens=1536,
        seed=0,
    )
    outputs = llm.generate(messages, sampling)

    saved_rows: list[dict] = []
    for idx, (meta, prompt, output) in enumerate(zip(prompt_rows, prompts, outputs)):
        text = (output.outputs[0].text or '').strip()
        saved_rows.append(
            {
                'idx': idx,
                'row_id': meta['row_id'],
                'problem': meta['problem'],
                'answer': meta['answer'],
                'trace_local_idx': meta['trace_local_idx'],
                'trace_text': meta['trace_text'],
                'prompt_text': prompt,
                'generated_principles': text,
                'word_count': len(text.split()),
            }
        )

    report = {
        'model_name': MODEL_NAME,
        'prompt_path': str(PROMPT_PATH.relative_to(REPO_ROOT)),
        'dataset_path': str(DATASET_PATH.relative_to(REPO_ROOT)),
        'split': 'test',
        'num_problems': len(source_rows),
        'num_trace_prompts': len(saved_rows),
        'mean_word_count': (sum(r['word_count'] for r in saved_rows) / len(saved_rows)) if saved_rows else 0.0,
        'max_model_len': 16384,
        'max_num_batched_tokens': 4096,
        'gpu_memory_utilization': 0.9,
        'temperature': 0.0,
        'max_tokens': 1536,
    }

    ROWS_PATH.write_text(json.dumps(saved_rows, indent=2), encoding='utf-8')
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding='utf-8')

    grouped: dict[int, list[dict]] = {}
    for row in saved_rows:
        grouped.setdefault(row['row_id'], []).append(row)

    md_lines: list[str] = []
    md_lines.append('# Principle Extractions By Problem')
    md_lines.append('')
    md_lines.append(f"- model: `{MODEL_NAME}`")
    md_lines.append(f"- prompt: `{PROMPT_PATH.relative_to(REPO_ROOT)}`")
    md_lines.append(f"- problems: {len(source_rows)}")
    md_lines.append(f"- trace prompts: {len(saved_rows)}")
    md_lines.append('')

    for row in source_rows:
        problem_rows = grouped.get(row['row_id'], [])
        md_lines.append(f"## Row {row['row_id']}")
        md_lines.append('')
        md_lines.append('**Problem**')
        md_lines.append(row['problem'].strip())
        md_lines.append('')
        for extracted in problem_rows:
            md_lines.append(f"### Correct Trace {extracted['trace_local_idx']}")
            md_lines.append('')
            md_lines.append('```text')
            md_lines.append(extracted['generated_principles'].rstrip())
            md_lines.append('```')
            md_lines.append('')
        md_lines.append('---')
        md_lines.append('')

    MARKDOWN_PATH.write_text('\n'.join(md_lines), encoding='utf-8')
    print(json.dumps(report, indent=2))
    print(MARKDOWN_PATH)


if __name__ == '__main__':
    main()
