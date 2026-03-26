from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from datasets import load_from_disk, Dataset, DatasetDict

REPO = Path('/workspace/action_abstraction')
ROOTS = [
    REPO / 'outputs/2026-03-21/contrastive_abstraction_prompting',
    REPO / 'outputs/2026-03-22/contrastive_abstraction_prompting/rlad_non_regressed_half_epoch_checkpoint_sweeps',
]

SWEEPS = []
# Earlier improved-only family
for name in [
    'rlad_sample12_qwen_baseline_vs_sft_sweep',
    'rlad_sample12_qwen_checkpoint_137_sweep',
    'rlad_sample12_qwen_checkpoint_274_sweep',
    'rlad_sample12_qwen_checkpoint_411_sweep',
]:
    SWEEPS.append(REPO / 'outputs/2026-03-21/contrastive_abstraction_prompting' / name)
# Later non-regressed family
for ckpt in ['0217','0434','0651','0868','1085','1302','1519','1736','1953','2170']:
    SWEEPS.append(REPO / 'outputs/2026-03-22/contrastive_abstraction_prompting/rlad_non_regressed_half_epoch_checkpoint_sweeps' / f'checkpoint_{ckpt}_sweep')


def load_rows(path: Path) -> list[dict[str, Any]]:
    ds = load_from_disk(str(path))
    if isinstance(ds, DatasetDict):
        ds = ds['train'] if 'train' in ds else next(iter(ds.values()))
    assert isinstance(ds, Dataset)
    return [dict(ds[i]) for i in range(len(ds))]


def word_count(s: str) -> int:
    return len(re.findall(r"\S+", s or ''))


def has_problem_numbers(abstraction: str) -> bool:
    return bool(re.search(r"\d", abstraction or ''))


def note_count(abstraction: str) -> int:
    return len(re.findall(r"<note\d+>|\[CORE\]|\[SUPPORT\]", abstraction or '', flags=re.I))


def description_len(abstraction: str) -> int:
    return word_count(abstraction)


def solution_len(sol: str) -> int:
    return word_count(sol)


def classify_row(row: dict[str, Any]) -> list[str]:
    tags = []
    abstraction = (row.get('generated_principles_text') or row.get('generated_abstraction') or '')
    sols = list(row.get('conditioned_generated_solution') or [])
    answers = list(row.get('conditioned_generated_answer') or [])
    baseline = float(row.get('additional_passrate', 0.0))
    conditioned = float(row.get('conditioned_passrate', 0.0))
    problem = row.get('problem') or ''
    gold = str(row.get('answer'))

    if not abstraction.strip():
        tags.append('empty_abstraction')
        return tags

    if has_problem_numbers(abstraction):
        tags.append('contains_concrete_numbers')

    wc = description_len(abstraction)
    if wc < 120:
        tags.append('too_thin_or_generic')
    if wc > 600:
        tags.append('overlong_abstraction')

    if note_count(abstraction) < 3:
        tags.append('underspecified_structure')

    lower_abs = abstraction.lower()
    if any(phrase in lower_abs for phrase in ['divisible by', 'last digit', 'alternating sum', 'count permutations', 'greedy', 'subset counting principle']):
        tags.append('heuristic_or_template_like')

    # Solution-side tags from traces.
    if sols:
        lens = [solution_len(s) for s in sols]
        if mean(lens) > 1800:
            tags.append('solver_trace_very_long')
        if any('Wait,' in s or 'wait,' in s for s in sols):
            tags.append('solver_self_correction_loop')
        if any('I need to' in s or 'Let me think' in s for s in sols):
            tags.append('solver_meta_reasoning_heavy')
        if any(s.count('Therefore') + s.count('therefore') > 15 for s in sols):
            tags.append('solver_overexplains')

    # Outcome tags.
    if conditioned < baseline:
        tags.append('regression')
    elif conditioned == baseline:
        tags.append('no_gain')
    elif conditioned > baseline:
        tags.append('improvement')

    if conditioned == 0.0:
        tags.append('all_four_solver_fail')
    elif 0.0 < conditioned < 1.0:
        tags.append('mixed_solver_outcomes')
    elif conditioned == 1.0:
        tags.append('all_four_solver_succeed')

    # Whether abstraction likely ignored: good baseline, similar solver drift not clear; use no-gain/regression with generic abstraction.
    if baseline >= 0.75 and conditioned < baseline and ('too_thin_or_generic' in tags or 'heuristic_or_template_like' in tags):
        tags.append('likely_unhelpful_generic_abstraction')

    if baseline == 0.0 and conditioned == 0.0 and wc > 200:
        tags.append('detailed_but_nonrescuing_abstraction')

    # Distinct wrong-answer diversity across solver samples.
    distinct_answers = len(set(a for a in answers if a))
    if distinct_answers >= 3 and conditioned < 1.0:
        tags.append('solver_answer_instability')
    elif distinct_answers == 1 and conditioned < 1.0 and answers:
        tags.append('consistent_wrong_answer')

    return tags

records = []
for sweep in SWEEPS:
    if not sweep.exists():
        continue
    sweep_name = sweep.name
    for temp_dir in sorted(sweep.glob('conditioned_temp_*')):
        ds_path = temp_dir / 'dataset'
        if not ds_path.exists():
            continue
        rows = load_rows(ds_path)
        temp = temp_dir.name.replace('conditioned_temp_', '')
        for row in rows:
            abstraction = (row.get('generated_principles_text') or row.get('generated_abstraction') or '')
            tags = classify_row(row)
            records.append({
                'sweep': sweep_name,
                'temp': temp,
                'source': row.get('source'),
                'sample_dataset_idx': row.get('sample_dataset_idx'),
                'baseline': float(row.get('additional_passrate', 0.0)),
                'conditioned': float(row.get('conditioned_passrate', 0.0)),
                'abstraction_idx': row.get('abstraction_idx'),
                'abstraction_wc': description_len(abstraction),
                'solution_wc_mean': mean([solution_len(s) for s in (row.get('conditioned_generated_solution') or [''])]),
                'abstraction': abstraction,
                'problem': row.get('problem'),
                'answer': row.get('answer'),
                'conditioned_generated_solution': row.get('conditioned_generated_solution') or [],
                'conditioned_generated_answer': row.get('conditioned_generated_answer') or [],
                'tags': tags,
            })

counter = Counter()
by_source = defaultdict(Counter)
for rec in records:
    for tag in rec['tags']:
        counter[tag] += 1
        by_source[rec['source']][tag] += 1

# Select representative examples for a focused subset of meaningful failure modes.
FOCUS = [
    'contains_concrete_numbers',
    'too_thin_or_generic',
    'overlong_abstraction',
    'likely_unhelpful_generic_abstraction',
    'detailed_but_nonrescuing_abstraction',
    'solver_self_correction_loop',
    'consistent_wrong_answer',
    'solver_answer_instability',
]

examples = {}
for tag in FOCUS:
    candidates = [r for r in records if tag in r['tags'] and r['source'] == 'aime']
    if not candidates:
        candidates = [r for r in records if tag in r['tags']]
    candidates.sort(key=lambda r: (r['conditioned'], -r['baseline'], r['sweep'], r['sample_dataset_idx'], r['abstraction_idx']))
    examples[tag] = candidates[:3]

out = {
    'num_records': len(records),
    'tag_counts': counter,
    'tag_counts_by_source': {k: dict(v) for k, v in by_source.items()},
    'examples': examples,
}

summary_path = REPO / 'outputs/2026-03-23/contrastive_abstraction_prompting/sft_failure_mode_review_summary.json'
summary_path.write_text(json.dumps(out, indent=2, default=str), encoding='utf-8')
print(summary_path)
print('records', len(records))
print('top_tags', counter.most_common(20))
