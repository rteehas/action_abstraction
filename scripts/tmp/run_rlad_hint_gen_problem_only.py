from __future__ import annotations
import json
import sys
from pathlib import Path

import torch
from datasets import load_from_disk, DatasetDict
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

REPO_ROOT = Path('/workspace/action_abstraction')
OUTPUT_DIR = REPO_ROOT / 'outputs/2026-03-19/contrastive_abstraction_eval/rlad_hint_gen_problem_only_r1'
DATASET_PATH = REPO_ROOT / 'contrastive_abstraction_datasets/deepscaler_mixed_correct_incorrect_256_all_correct'
PROMPT_PATH = REPO_ROOT / 'prompt_templates/rlad_abstraction_generation_prompt_template.txt'
MODEL_NAME = 'CMU-AIRe/RLAD-Hint-Gen'


def render_prompt(template: str, problem: str) -> str:
    prompt = template
    prompt = prompt.replace('{{PROBLEM}}', problem)
    prompt = prompt.replace('{problem_description}', problem)
    return prompt


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset = load_from_disk(str(DATASET_PATH))
    if isinstance(dataset, DatasetDict):
        ds = dataset['test']
    else:
        ds = dataset
    rows = [dict(ds[i]) for i in range(min(26, ds.num_rows))]

    template = PROMPT_PATH.read_text(encoding='utf-8')
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    prompts = [render_prompt(template, row['problem']) for row in rows]
    messages = [
        tokenizer.apply_chat_template(
            [{'role': 'user', 'content': prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for prompt in prompts
    ]

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
        max_tokens=4096,
        seed=0,
    )
    outputs = llm.generate(messages, sampling)

    saved_rows = []
    for idx, (row, output, prompt) in enumerate(zip(rows, outputs, prompts)):
        text = output.outputs[0].text.strip()
        saved_rows.append({
            'idx': idx,
            'row_id': row['row_id'],
            'problem': row['problem'],
            'answer': row['answer'],
            'prompt_text': prompt,
            'generated_text': text,
            'word_count': len(text.split()),
        })

    report = {
        'num_rows': len(saved_rows),
        'model_name': MODEL_NAME,
        'prompt_path': str(PROMPT_PATH.relative_to(REPO_ROOT)),
        'dataset_path': str(DATASET_PATH.relative_to(REPO_ROOT)),
        'split': 'test',
        'max_model_len': 16384,
        'max_num_batched_tokens': 4096,
        'gpu_memory_utilization': 0.9,
        'temperature': 0.0,
        'max_tokens': 4096,
        'mean_word_count': (sum(r['word_count'] for r in saved_rows) / len(saved_rows)) if saved_rows else 0.0,
    }

    (OUTPUT_DIR / 'rows.json').write_text(json.dumps(saved_rows, indent=2), encoding='utf-8')
    (OUTPUT_DIR / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
