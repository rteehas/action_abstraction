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

from contrastive_abstraction_utils import format_solution_block, read_text, strip_think_blocks
from process_deepscaler_dataset import extract_answer_from_solution, parse, verify

DATASET_PATH = REPO_ROOT / 'contrastive_abstraction_datasets/deepscaler_mixed_correct_incorrect_256_all_correct'
PROMPT_PATH = REPO_ROOT / 'prompt_templates/rlad_abstraction_generation_prompt_template_correct_traces.txt'
SOLVER_PROMPT_PATH = REPO_ROOT / 'prompt_templates/hint_conditioned_problem_solving_rich_v1.txt'
OUTPUT_DIR = REPO_ROOT / 'outputs/2026-03-19/contrastive_abstraction_eval/rlad_correct_traces_raw_prompt_solver_rich_r1'
BASE_MODEL = 'Qwen/Qwen3-1.7B'


def render_prompt(prompt_template: str, ex: dict) -> str:
    correct_block = format_solution_block('Correct Trace', ex['selected_correct_solutions'], max_chars=6000)
    prompt = prompt_template
    prompt = prompt.replace('{{PROBLEM}}', ex['problem'])
    prompt = prompt.replace('{{CORRECT_SOLUTIONS_BLOCK}}', correct_block)
    prompt = prompt.replace('{problem_description}', ex['problem'])
    prompt = prompt.replace('{correct_solutions}', correct_block)
    return prompt


def make_solver_prompt(problem: str, abstraction: str, template: str) -> str:
    return template.replace('{{PROBLEM}}', problem).replace('{{ABSTRACTION}}', abstraction)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset = load_from_disk(str(DATASET_PATH))
    if isinstance(dataset, DatasetDict):
        ds = dataset['test']
    else:
        ds = dataset
    rows = [dict(ds[i]) for i in range(min(26, ds.num_rows))]

    prompt_template = read_text(PROMPT_PATH)
    solver_template = read_text(SOLVER_PROMPT_PATH)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    abstraction_prompts = [render_prompt(prompt_template, ex) for ex in rows]
    abstraction_messages = [
        tokenizer.apply_chat_template(
            [{'role': 'user', 'content': prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for prompt in abstraction_prompts
    ]

    print(f'Device name = {torch.cuda.get_device_name(0)}')
    llm = LLM(
        model=BASE_MODEL,
        enable_lora=False,
        max_model_len=20480,
        max_num_batched_tokens=4096,
        gpu_memory_utilization=0.9,
    )

    abstraction_sampling = SamplingParams(
        temperature=1.0,
        top_p=0.95,
        top_k=20,
        max_tokens=4096,
        seed=0,
    )
    abstraction_outputs = llm.generate(abstraction_messages, abstraction_sampling)
    raw_abstractions = [strip_think_blocks(out.outputs[0].text or '').strip() for out in abstraction_outputs]

    solver_prompts = []
    solver_indices = []
    for idx, ex in enumerate(rows):
        abstraction = raw_abstractions[idx]
        if not abstraction:
            continue
        solver_prompts.append(
            tokenizer.apply_chat_template(
                [{'role': 'user', 'content': make_solver_prompt(ex['problem'], abstraction, solver_template)}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
        )
        solver_indices.append(idx)

    solver_sampling = SamplingParams(
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        max_tokens=2048,
        seed=0,
    )
    solver_outputs = [None] * len(rows)
    predicted_answers = [None] * len(rows)
    solver_correct = [False] * len(rows)
    if solver_prompts:
        solver_generations = llm.generate(solver_prompts, solver_sampling)
        for prompt_idx, dataset_idx in enumerate(solver_indices):
            text = solver_generations[prompt_idx].outputs[0].text
            solver_outputs[dataset_idx] = text
            boxed = extract_answer_from_solution(text)
            predicted_answers[dataset_idx] = boxed
            target = f"\\boxed{{{rows[dataset_idx]['answer']}}}"
            if boxed is not None:
                try:
                    solver_correct[dataset_idx] = bool(verify(parse(target), parse(boxed)))
                except Exception:
                    solver_correct[dataset_idx] = False

    saved_rows = []
    for idx, ex in enumerate(rows):
        saved_rows.append({
            'row_id': ex['row_id'],
            'problem': ex['problem'],
            'answer': ex['answer'],
            'baseline_passrate': ex['passrate'],
            'baseline_num_correct': ex['num_correct'],
            'selected_correct_indices': ex['selected_correct_indices'],
            'selected_incorrect_indices': ex['selected_incorrect_indices'],
            'abstraction_prompt_text': abstraction_prompts[idx],
            'generated_abstraction_text': abstraction_outputs[idx].outputs[0].text,
            'abstraction': raw_abstractions[idx],
            'solver_output_text': solver_outputs[idx],
            'solver_predicted_answer': predicted_answers[idx],
            'solver_correct': solver_correct[idx],
        })

    num_rows = len(saved_rows)
    num_correct = sum(1 for row in saved_rows if row['solver_correct'])
    num_valid_answers = sum(1 for row in saved_rows if row['solver_predicted_answer'] is not None)
    baseline_passrate_mean = sum(row['baseline_passrate'] for row in saved_rows) / num_rows if num_rows else 0.0
    baseline_num_correct_mean = sum(row['baseline_num_correct'] for row in saved_rows) / num_rows if num_rows else 0.0
    report = {
        'num_rows': num_rows,
        'num_correct': num_correct,
        'accuracy': num_correct / num_rows if num_rows else 0.0,
        'num_valid_answers': num_valid_answers,
        'baseline_passrate_mean': baseline_passrate_mean,
        'baseline_num_correct_mean': baseline_num_correct_mean,
        'absolute_accuracy_lift': (num_correct / num_rows if num_rows else 0.0) - baseline_passrate_mean,
        'relative_accuracy_lift': (((num_correct / num_rows) - baseline_passrate_mean) / baseline_passrate_mean) if num_rows and baseline_passrate_mean else None,
        'title': 'RLAD Correct Traces Raw Prompt Evaluation',
        'dataset_path': str(DATASET_PATH.relative_to(REPO_ROOT)),
        'split': 'test',
        'base_model': BASE_MODEL,
        'abstraction_prompt_path': str(PROMPT_PATH.relative_to(REPO_ROOT)),
        'solver_prompt_path': str(SOLVER_PROMPT_PATH.relative_to(REPO_ROOT)),
        'raw_output_used_as_abstraction': True,
        'max_model_len': 20480,
        'max_num_batched_tokens': 4096,
        'gpu_memory_utilization': 0.9,
    }

    (OUTPUT_DIR / 'rows.json').write_text(json.dumps(saved_rows, indent=2), encoding='utf-8')
    (OUTPUT_DIR / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
