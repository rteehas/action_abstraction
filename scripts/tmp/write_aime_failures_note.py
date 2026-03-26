from datasets import load_from_disk
from pathlib import Path
import sys

repo = Path('/workspace/action_abstraction')
if str(repo) not in sys.path:
    sys.path.insert(0, str(repo))
from process_deepscaler_dataset import parse, verify

root = repo / 'outputs/2026-03-22/contrastive_abstraction_prompting/rlad_top4_rlad_solver_template_sweep'
path = root / 'checkpoint_1953_sweep/conditioned_temp_0p0/dataset'
out = root / 'final_report/AIME_FAILURE_CASES_checkpoint_1953_temp0p0.md'

ds = load_from_disk(str(path))

seen = set()
selected = []
for row in ds:
    if row['source'] != 'aime':
        continue
    key = row['sample_dataset_idx']
    if key in seen:
        continue
    if float(row['conditioned_passrate']) >= 1.0:
        continue
    seen.add(key)
    selected.append(row)

def is_correct(expected_answer: str, predicted_boxed_answer: str | None) -> bool:
    if predicted_boxed_answer is None:
        return False
    target = f'\\boxed{{{expected_answer}}}'
    try:
        return bool(verify(parse(target), parse(predicted_boxed_answer)))
    except Exception:
        return False

lines = [
    '# AIME Failure Cases',
    '',
    'Representative failed AIME examples from the RLAD-template rerun using the best average checkpoint setting:',
    '- checkpoint: `checkpoint-1953`',
    '- abstraction temperature: `0.0`',
    '- solver prompt: `prompt_templates/rlad_solver_template.txt`',
    '- abstraction inserted as: `cheatsheet`',
    '- baseline in this rerun: `AIME = 0.4167`, `overall = 0.6667`',
    '',
    'For each problem below, this note shows one representative abstraction row, the generated abstraction, and one full incorrect conditioned solution trace.',
]

for row in selected:
    answers = list(row.get('conditioned_generated_answer') or [])
    sols = list(row.get('conditioned_generated_solution') or [])
    bad_idx = None
    for i, ans in enumerate(answers):
        if not is_correct(str(row['answer']), ans):
            bad_idx = i
            break
    if bad_idx is None:
        bad_idx = 0
    pred = answers[bad_idx] if bad_idx < len(answers) else None
    sol = sols[bad_idx] if bad_idx < len(sols) else ''
    lines += [
        '',
        f"## Problem {row['sample_dataset_idx']}",
        '',
        f"- source: `{row['source']}`",
        f"- gold answer: `{row['answer']}`",
        f"- baseline passrate: `{float(row['additional_passrate']):.2f}`",
        f"- conditioned passrate: `{float(row['conditioned_passrate']):.2f}`",
        f"- abstraction idx: `{row.get('abstraction_idx')}`",
        f"- shown solver sample: `{bad_idx}`",
        f"- shown predicted answer: `{pred}`",
        '',
        '### Problem',
        '',
        row['problem'],
        '',
        '### Generated Abstraction',
        '',
        '```text',
        (row.get('generated_principles_text') or row.get('generated_abstraction') or '').rstrip(),
        '```',
        '',
        '### Full Incorrect Conditioned Solution Trace',
        '',
        '```text',
        sol.rstrip(),
        '```',
    ]

out.write_text('\n'.join(lines) + '\n', encoding='utf-8')
print(out)
print(f'rows={len(selected)}')
