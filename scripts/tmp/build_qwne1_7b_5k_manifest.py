from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

from datasets import load_from_disk

REPO_ROOT = Path('/workspace/action_abstraction')
SCRIPTS_DIR = REPO_ROOT / 'scripts'
for path in (REPO_ROOT, SCRIPTS_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from contrastive_abstraction_utils import classify_generated_answers, strip_think_blocks

DATASET_PATH = Path('/deepscaler_qwne1_7B_solutions_scored')
OUTPUT_DIR = REPO_ROOT / 'outputs/2026-03-20/contrastive_abstraction_prompting/principle_v5_qwne1_7b_scored_5k_eval'
MANIFEST_PATH = OUTPUT_DIR / 'subset5k_manifest.json'
SUMMARY_PATH = OUTPUT_DIR / 'subset5k_manifest_summary.json'
SEED = 20260320
SUBSET_SIZE = 5000


def make_row_payload(idx: int, row: dict, first_correct_idx: int, first_correct_trace: str) -> dict:
    return {
        'row_id': idx,
        'problem': row['problem'],
        'answer': row['answer'],
        'passrate': float(row['passrate']),
        'num_correct': int(row['num_correct']),
        'first_correct_trace_index': int(first_correct_idx),
        'first_correct_trace_text': first_correct_trace,
        'source_dataset_path': str(DATASET_PATH),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ds = load_from_disk(str(DATASET_PATH))

    eligible_rows: list[dict] = []
    skip_reasons = Counter()
    passrate_counts = Counter()
    eligible_passrate_counts = Counter()

    for idx, row in enumerate(ds):
        passrate = float(row['passrate'])
        passrate_counts[passrate] += 1
        if passrate <= 0:
            skip_reasons['passrate_le_zero'] += 1
            continue

        generated_solutions = row.get('generated_solution') or []
        generated_answers = row.get('generated_answer') or []
        if not generated_solutions or not generated_answers:
            skip_reasons['missing_generated_fields'] += 1
            continue

        correctness = classify_generated_answers(row['answer'], generated_answers)
        first_correct_idx = None
        first_correct_trace = None
        for candidate_idx, (is_correct, trace_text) in enumerate(zip(correctness, generated_solutions)):
            if not is_correct:
                continue
            cleaned = strip_think_blocks(trace_text or '')
            if not cleaned:
                continue
            first_correct_idx = candidate_idx
            first_correct_trace = cleaned
            break

        if first_correct_idx is None or first_correct_trace is None:
            skip_reasons['no_nonempty_verified_correct_trace'] += 1
            continue

        eligible_rows.append(make_row_payload(idx, row, first_correct_idx, first_correct_trace))
        eligible_passrate_counts[passrate] += 1

    if len(eligible_rows) < SUBSET_SIZE:
        raise ValueError(f'Only found {len(eligible_rows)} eligible rows, need {SUBSET_SIZE}')

    rng = random.Random(SEED)
    sampled_rows = rng.sample(eligible_rows, SUBSET_SIZE)
    sampled_rows.sort(key=lambda row: row['row_id'])

    sampled_passrate_counts = Counter(row['passrate'] for row in sampled_rows)
    manifest = {
        'metadata': {
            'title': 'QWNE1.7B scored dataset 5K principle-eval subset',
            'dataset_path': str(DATASET_PATH),
            'subset_size': SUBSET_SIZE,
            'selection': 'deterministic random sample from eligible rows with passrate > 0 and non-empty first verified correct generated trace',
            'selection_seed': SEED,
            'candidate_prompt_path': 'prompt_templates/principle_extraction_template_v5.txt',
            'solver_seeds': [1001, 1002, 1003, 1004],
        },
        'train_rows': [],
        'val_rows': sampled_rows,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding='utf-8')

    summary = {
        'dataset_length': len(ds),
        'passrate_counts_all_rows': {str(k): int(v) for k, v in sorted(passrate_counts.items())},
        'skip_reasons': dict(skip_reasons),
        'eligible_rows': len(eligible_rows),
        'eligible_passrate_counts': {str(k): int(v) for k, v in sorted(eligible_passrate_counts.items())},
        'sampled_rows': len(sampled_rows),
        'sampled_passrate_counts': {str(k): int(v) for k, v in sorted(sampled_passrate_counts.items())},
        'sampled_row_ids_head': [row['row_id'] for row in sampled_rows[:20]],
        'selection_seed': SEED,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
