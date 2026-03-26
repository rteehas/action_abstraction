from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

import torch
from datasets import Dataset, DatasetDict, load_from_disk
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

REPO_ROOT = Path('/workspace/action_abstraction')
SCRIPTS_DIR = REPO_ROOT / 'scripts'
for path in (REPO_ROOT, SCRIPTS_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from contrastive_abstraction_utils import format_solution_block, read_text, strip_think_blocks
from eval_principle_extraction_prompt import generate_texts, score_solver_output

BASE_MODEL = 'Qwen/Qwen3-1.7B'
PRINCIPLE_PROMPT_PATH = REPO_ROOT / 'prompt_templates/principle_extraction_template_v5.txt'
SOLVER_PROMPT_PATH = REPO_ROOT / 'prompt_templates/hint_conditioned_problem_solving_rich_v1.txt'
SPLIT_MANIFEST_PATH = REPO_ROOT / 'outputs/2026-03-20/contrastive_abstraction_prompting/gepa_principle_extraction_v5_128_32_balanced/split_manifest.json'
OUTPUT_DIR = REPO_ROOT / 'outputs/2026-03-20/contrastive_abstraction_prompting/principle_v5_first_vs_all_correct_val32_quicktest'
PRINCIPLE_TEMPERATURE = 0.0
SOLVER_TEMPERATURE = 0.6
SOLVER_SEEDS = [1001, 1002, 1003, 1004]
MAX_TOKENS = 4096
MAX_SOLUTION_CHARS = 6000
MAX_MODEL_LEN = 32768
MAX_NUM_BATCHED_TOKENS = 4096
GPU_MEMORY_UTILIZATION = 0.9
PROMPT_BATCH_SIZE = 64


def load_dataset_dict(path: str) -> DatasetDict:
    loaded = load_from_disk(path)
    if isinstance(loaded, DatasetDict):
        return loaded
    return DatasetDict({'train': loaded})


def load_source_examples(dataset_path: str) -> dict[int, dict[str, Any]]:
    dataset_dict = load_dataset_dict(str(REPO_ROOT / dataset_path))
    by_row_id: dict[int, dict[str, Any]] = {}
    for split_name, ds in dataset_dict.items():
        assert isinstance(ds, Dataset)
        for idx, ex in enumerate(ds):
            row_id = int(ex.get('row_id', idx))
            payload = dict(ex)
            payload['source_split'] = split_name
            by_row_id[row_id] = payload
    return by_row_id


def ensure_trace_pairs(example: dict[str, Any]) -> tuple[list[tuple[int, str]], list[tuple[int, str]], list[bool]]:
    generated_solutions = example.get('generated_solution')
    generated_answers = example.get('generated_answer')
    if generated_solutions is not None and generated_answers is not None:
        solutions = list(generated_solutions)
        if example.get('solution_correctness') is not None:
            correctness = [bool(value) for value in example['solution_correctness']]
        else:
            raise ValueError('solution_correctness is required when generated_solution/generated_answer are present')
        correct_pairs = [(idx, solutions[idx]) for idx, is_correct in enumerate(correctness) if is_correct]
        incorrect_pairs = [(idx, solutions[idx]) for idx, is_correct in enumerate(correctness) if not is_correct]
        return correct_pairs, incorrect_pairs, correctness

    correct_indices = list(example.get('selected_correct_indices') or range(len(example.get('selected_correct_solutions') or [])))
    incorrect_indices = list(example.get('selected_incorrect_indices') or range(len(example.get('selected_incorrect_solutions') or [])))
    correct_solutions = list(example.get('selected_correct_solutions') or [])
    incorrect_solutions = list(example.get('selected_incorrect_solutions') or [])
    correct_pairs = list(zip(correct_indices, correct_solutions))
    incorrect_pairs = list(zip(incorrect_indices, incorrect_solutions))
    correctness = [True] * len(correct_pairs) + [False] * len(incorrect_pairs)
    return correct_pairs, incorrect_pairs, correctness


def cleaned_correct_traces(example: dict[str, Any]) -> list[dict[str, Any]]:
    correct_pairs, _, _ = ensure_trace_pairs(example)
    traces: list[dict[str, Any]] = []
    for trace_idx, trace_text in correct_pairs:
        cleaned = strip_think_blocks(trace_text)
        if not cleaned:
            continue
        traces.append({'trace_index': int(trace_idx), 'text': cleaned})
    return traces


def render_principle_prompt(problem: str, traces: list[str], template: str) -> str:
    correct_block = format_solution_block('Correct Trace', traces, max_chars=MAX_SOLUTION_CHARS)
    return template.replace('{{PROBLEM}}', problem).replace('{{CORRECT_SOLUTIONS_BLOCK}}', correct_block)


def evaluate_variant(
    llm: LLM,
    tokenizer: AutoTokenizer,
    rows: list[dict[str, Any]],
    candidate_template: str,
    solver_template: str,
    trace_mode: str,
) -> dict[str, Any]:
    variant_rows: list[dict[str, Any]] = []
    principle_prompts: list[str] = []

    for row in rows:
        traces = row['all_correct_traces_text'] if trace_mode == 'all_correct' else [row['first_correct_trace_text']]
        variant_row = dict(row)
        variant_row['trace_mode'] = trace_mode
        variant_row['used_correct_trace_count'] = len(traces)
        principle_prompt = render_principle_prompt(row['problem'], traces, candidate_template)
        variant_row['principle_prompt_text'] = principle_prompt
        variant_rows.append(variant_row)
        principle_prompts.append(principle_prompt)

    principle_sampling = SamplingParams(
        temperature=PRINCIPLE_TEMPERATURE,
        top_p=1.0,
        top_k=-1,
        max_tokens=MAX_TOKENS,
        seed=0,
    )
    principle_outputs = generate_texts(
        llm,
        tokenizer,
        principle_prompts,
        principle_sampling,
        batch_size=PROMPT_BATCH_SIZE,
        enable_thinking=False,
    )

    solver_prompts: list[str] = []
    prompt_row_indices: list[int] = []
    for row_idx, (row, principle_text) in enumerate(zip(variant_rows, principle_outputs)):
        row['generated_principles_text'] = principle_text
        row['principles'] = principle_text.strip()
        row['solver_samples'] = []
        if not row['principles']:
            row['mean_solver_accuracy'] = 0.0
            continue
        solver_prompts.append(
            solver_template.replace('{{PROBLEM}}', row['problem']).replace('{{ABSTRACTION}}', row['principles'])
        )
        prompt_row_indices.append(row_idx)

    for solver_seed in SOLVER_SEEDS:
        solver_sampling = SamplingParams(
            temperature=SOLVER_TEMPERATURE,
            top_p=0.95,
            top_k=20,
            max_tokens=MAX_TOKENS,
            seed=solver_seed,
        )
        solver_outputs = generate_texts(
            llm,
            tokenizer,
            solver_prompts,
            solver_sampling,
            batch_size=PROMPT_BATCH_SIZE,
            enable_thinking=False,
        )
        for row_idx, output_text in zip(prompt_row_indices, solver_outputs):
            predicted_answer, correct = score_solver_output(variant_rows[row_idx]['answer'], output_text)
            variant_rows[row_idx]['solver_samples'].append(
                {
                    'seed': solver_seed,
                    'output_text': output_text,
                    'predicted_answer': predicted_answer,
                    'correct': correct,
                }
            )

    for row in variant_rows:
        samples = row.get('solver_samples') or []
        row['mean_solver_accuracy'] = (
            sum(1.0 for sample in samples if sample.get('correct')) / len(samples) if samples else 0.0
        )

    score_counter = Counter(float(row['mean_solver_accuracy']) for row in variant_rows)
    report = {
        'trace_mode': trace_mode,
        'num_rows': len(variant_rows),
        'val_accuracy': mean(float(row['mean_solver_accuracy']) for row in variant_rows) if variant_rows else 0.0,
        'baseline_passrate_mean': mean(float(row['passrate']) for row in variant_rows) if variant_rows else 0.0,
        'mean_principle_words': mean(len((row.get('principles') or '').split()) for row in variant_rows) if variant_rows else 0.0,
        'score_distribution': {str(k): int(v) for k, v in sorted(score_counter.items())},
    }
    return {'rows': variant_rows, 'report': report}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(SPLIT_MANIFEST_PATH.read_text())
    source_examples = load_source_examples(manifest['source_dataset_path'])
    candidate_template = read_text(PRINCIPLE_PROMPT_PATH)
    solver_template = read_text(SOLVER_PROMPT_PATH)

    base_rows: list[dict[str, Any]] = []
    for row in manifest['val_rows']:
        row_id = int(row['row_id'])
        example = source_examples[row_id]
        all_correct = cleaned_correct_traces(example)
        if not all_correct:
            continue
        base_rows.append(
            {
                **row,
                'all_correct_trace_indices': [item['trace_index'] for item in all_correct],
                'all_correct_traces_text': [item['text'] for item in all_correct],
                'all_correct_trace_count': len(all_correct),
            }
        )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    print(f'Device name = {torch.cuda.get_device_name(0)}')
    llm = LLM(
        model=BASE_MODEL,
        enable_lora=False,
        max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        max_model_len=MAX_MODEL_LEN,
    )

    first_result = evaluate_variant(llm, tokenizer, base_rows, candidate_template, solver_template, 'first_correct')
    all_result = evaluate_variant(llm, tokenizer, base_rows, candidate_template, solver_template, 'all_correct')

    first_by_id = {int(row['row_id']): row for row in first_result['rows']}
    all_by_id = {int(row['row_id']): row for row in all_result['rows']}
    comparison_rows: list[dict[str, Any]] = []
    delta_counter = Counter()
    better_all = 0
    better_first = 0
    equal = 0
    for row in base_rows:
        row_id = int(row['row_id'])
        first_score = float(first_by_id[row_id]['mean_solver_accuracy'])
        all_score = float(all_by_id[row_id]['mean_solver_accuracy'])
        delta = round(all_score - first_score, 6)
        delta_counter[delta] += 1
        if delta > 0:
            better_all += 1
        elif delta < 0:
            better_first += 1
        else:
            equal += 1
        comparison_rows.append(
            {
                'row_id': row_id,
                'passrate': float(row['passrate']),
                'num_correct': int(row['num_correct']),
                'all_correct_trace_count': int(row['all_correct_trace_count']),
                'first_correct_trace_index': row.get('first_correct_trace_index'),
                'all_correct_trace_indices': row['all_correct_trace_indices'],
                'first_correct_score': first_score,
                'all_correct_score': all_score,
                'delta_all_minus_first': delta,
                'first_principles': first_by_id[row_id]['principles'],
                'all_principles': all_by_id[row_id]['principles'],
            }
        )

    summary = {
        'title': 'Quick comparison: first correct trace vs all correct traces',
        'split_manifest_path': str(SPLIT_MANIFEST_PATH),
        'candidate_prompt_path': str(PRINCIPLE_PROMPT_PATH),
        'solver_prompt_path': str(SOLVER_PROMPT_PATH),
        'base_model': BASE_MODEL,
        'num_rows': len(base_rows),
        'solver_seeds': SOLVER_SEEDS,
        'first_correct': first_result['report'],
        'all_correct': all_result['report'],
        'all_minus_first_val_accuracy': all_result['report']['val_accuracy'] - first_result['report']['val_accuracy'],
        'rows_better_with_all_correct': better_all,
        'rows_better_with_first_correct': better_first,
        'rows_equal': equal,
        'delta_distribution': {str(k): int(v) for k, v in sorted(delta_counter.items())},
        'avg_all_correct_trace_count': mean(row['all_correct_trace_count'] for row in base_rows) if base_rows else 0.0,
    }

    (OUTPUT_DIR / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    (OUTPUT_DIR / 'comparison_rows.json').write_text(json.dumps(comparison_rows, indent=2), encoding='utf-8')
    (OUTPUT_DIR / 'first_correct_rows.json').write_text(json.dumps(first_result['rows'], indent=2), encoding='utf-8')
    (OUTPUT_DIR / 'all_correct_rows.json').write_text(json.dumps(all_result['rows'], indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
