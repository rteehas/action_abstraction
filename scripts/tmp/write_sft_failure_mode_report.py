from __future__ import annotations

import json
from pathlib import Path

REPO = Path('/workspace/action_abstraction')
summary_path = REPO / 'outputs/2026-03-23/contrastive_abstraction_prompting/sft_failure_mode_review_summary.json'
out_path = REPO / 'outputs/2026-03-23/contrastive_abstraction_prompting/SFT_FAILURE_MODE_REPORT.md'

data = json.loads(summary_path.read_text(encoding='utf-8'))
examples = data['examples']
counts = data['tag_counts']
by_source = data['tag_counts_by_source']

FOCUS = [
    ('contains_concrete_numbers', 'Problem-Specific Abstractions', 'Many abstractions directly mention concrete values, expressions, or named objects from the underlying problem. That violates the intended RLAD abstraction behavior and often drags the solver back into brittle task-specific reasoning instead of reusable structure.'),
    ('overlong_abstraction', 'Overlong And Redundant Abstractions', 'Some abstractions are so long and repetitive that they dilute the actual lever. The solver then spends most of its reasoning budget unpacking or arguing with the abstraction rather than using it cleanly.'),
    ('heuristic_or_template_like', 'Template-Like Heuristics', 'Another common failure is the opposite: an abstraction that reads like a generic canned heuristic. These often mention broad tactics like divisibility, greedy choice, or counting but never isolate the decisive reduction for the specific problem class.'),
    ('detailed_but_nonrescuing_abstraction', 'Detailed But Non-Rescuing Abstractions', 'A large subset of failed AIME rows have detailed abstractions that look mathematically serious but still do not rescue zero-baseline problems. These typically gesture at the right domain but encode the wrong reduction or omit the crux needed to close the proof.'),
    ('consistent_wrong_answer', 'Consistent Wrong-Answer Lock-In', 'Sometimes the abstraction is wrong in a very specific way, and all four solver samples converge to the same wrong answer or nearly the same path. That is a stronger failure than noisy solving because it means the abstraction actively channels the solver into one bad derivation.'),
    ('solver_answer_instability', 'Solver Instability Under One Abstraction', 'Other times the abstraction is not wrong enough to fully lock the solver, but it is still too vague or internally inconsistent. Then the four solver samples scatter across different incorrect answers, which usually means the abstraction did not pin down a stable productive reduction.'),
]

lines = [
    '# SFT Failure Mode Review',
    '',
    'This note reviews the RLAD sample-12 evaluation outputs from the SFT checkpoint sweeps. I scanned all abstraction-conditioned rows from both SFT families:',
    '- earlier improved-only family: `checkpoint-137`, `checkpoint-274`, `checkpoint-411`, plus the final `baseline_vs_sft` checkpoint run',
    '- later non-regressed half-epoch family: `checkpoint-0217` through `checkpoint-2170`',
    '',
    'Total reviewed rows: `{}` abstraction-conditioned rows, corresponding to all stored abstraction outputs and all stored conditioned solver traces from those sweeps.'.format(data['num_records']),
    '',
    '## High-Level Counts',
    '',
    '- AIME rows are much more failure-heavy than AMC rows.',
    '- `contains_concrete_numbers`: {} total, AIME {}, AMC {}.'.format(counts.get('contains_concrete_numbers', 0), by_source.get('aime', {}).get('contains_concrete_numbers', 0), by_source.get('amc', {}).get('contains_concrete_numbers', 0)),
    '- `detailed_but_nonrescuing_abstraction`: {} total, almost entirely AIME {}.'.format(counts.get('detailed_but_nonrescuing_abstraction', 0), by_source.get('aime', {}).get('detailed_but_nonrescuing_abstraction', 0)),
    '- `solver_answer_instability`: {} total, AIME {}, AMC {}.'.format(counts.get('solver_answer_instability', 0), by_source.get('aime', {}).get('solver_answer_instability', 0), by_source.get('amc', {}).get('solver_answer_instability', 0)),
    '- `consistent_wrong_answer`: {} total, AIME {}, AMC {}.'.format(counts.get('consistent_wrong_answer', 0), by_source.get('aime', {}).get('consistent_wrong_answer', 0), by_source.get('amc', {}).get('consistent_wrong_answer', 0)),
    '- `regression`: {} total, compared with `improvement`: {}.'.format(counts.get('regression', 0), counts.get('improvement', 0)),
    '',
    'Two tags were nearly ubiquitous and are not especially diagnostic by themselves: `solver_self_correction_loop` and `solver_meta_reasoning_heavy`. They mostly reflect the long-form Qwen reasoning style rather than a distinct abstraction failure class.',
]

for tag, title, desc in FOCUS:
    lines.extend(['', f'## {title}', '', desc, ''])
    lines.append('- Count: `{}`'.format(counts.get(tag, 0)))
    if tag in by_source.get('aime', {}) or tag in by_source.get('amc', {}):
        lines.append('- By source: `AIME={}`, `AMC={}`'.format(by_source.get('aime', {}).get(tag, 0), by_source.get('amc', {}).get(tag, 0)))
    exs = examples.get(tag, [])
    if not exs:
        lines.append('- No representative examples were selected for this tag.')
        continue
    for i, ex in enumerate(exs, start=1):
        abs_text = ex['abstraction'].strip()
        if len(abs_text) > 1800:
            abs_text = abs_text[:1800].rstrip() + '\n...'
        bad_trace = ''
        sols = ex.get('conditioned_generated_solution') or []
        if sols:
            bad_trace = sols[0].strip()
            if len(bad_trace) > 2200:
                bad_trace = bad_trace[:2200].rstrip() + '\n...'
        answers = ex.get('conditioned_generated_answer') or []
        lines.extend([
            '',
            '### Example {}'.format(i),
            '',
            '- sweep: `{}`'.format(ex['sweep']),
            '- abstraction temperature: `{}`'.format(ex['temp']),
            '- source/problem id: '`{}` / `{}`'.format(ex['source'], ex['sample_dataset_idx']),
            '- baseline -> conditioned: `{:.2f} -> {:.2f}`'.format(ex['baseline'], ex['conditioned']),
            '- abstraction idx: `{}`'.format(ex['abstraction_idx']),
            '- abstraction words: `{}`'.format(ex['abstraction_wc']),
            '- mean solver-trace words: `{:.1f}`'.format(ex['solution_wc_mean']),
            '- gold answer: `{}`'.format(ex['answer']),
            '- generated answers: `{}`'.format(answers),
            '',
            'Problem:',
            ex['problem'],
            '',
            'Abstraction excerpt:',
            '```text',
            abs_text,
            '```',
        ])
        if bad_trace:
            lines.extend([
                '',
                'Conditioned solution trace excerpt:',
                '```text',
                bad_trace,
                '```',
            ])

lines.extend([
    '',
    '## Synthesis',
    '',
    'The dominant failure pattern is not empty output. It is mispackaged signal. The SFT models frequently produce abstractions that are either too concrete, too verbose, or too generic, and the solver then either overfits to one wrong derivation or flails among several wrong ones.',
    '',
    'AIME failures are dominated by two regimes:',
    '- structurally detailed but ultimately wrong abstractions that never rescue zero-baseline problems',
    '- abstractions that leak concrete problem content and induce brittle derivations',
    '',
    'AMC is much healthier overall, but the main residual failure mode there is regression on already-solvable rows from abstractions that are generic or slightly misdirecting rather than catastrophically wrong.',
    '',
    'Operationally, the next improvement target is not just better average abstraction quality. It is better abstraction compression: keep the decisive reduction, strip concrete values, and avoid multi-paragraph derivations masquerading as reusable hints.',
])

out_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
print(out_path)
