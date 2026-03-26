import json
from pathlib import Path

rows_path = Path('/workspace/action_abstraction/outputs/2026-03-20/contrastive_abstraction_prompting/gepa_principle_extraction_v5_128_32_balanced/candidate_evals/8ba9e6b438e4c909/rows.json')
out_path = Path('/workspace/action_abstraction/outputs/2026-03-20/contrastive_abstraction_prompting/gepa_principle_extraction_v5_128_32_balanced/VAL_PASSRATE_REGRESSIONS_v5_SEED.md')

rows = json.loads(rows_path.read_text(encoding='utf-8'))
regressions = [
    row for row in rows
    if row.get('split') == 'val' and float(row.get('mean_solver_accuracy', 0.0)) < float(row.get('passrate', 0.0))
]
regressions.sort(key=lambda row: (float(row['passrate']), float(row['mean_solver_accuracy']), int(row['row_id'])))

lines = []
lines.append('# Validation Rows Where Seed `principle_extraction_template_v5` Underperforms Original Passrate')
lines.append('')
lines.append('This note collects every validation row from the seed `principle_extraction_template_v5.txt` run where the principle-conditioned solve rate was lower than the row\'s original source passrate.')
lines.append('')
lines.append('Run root:')
lines.append('- `outputs/2026-03-20/contrastive_abstraction_prompting/gepa_principle_extraction_v5_128_32_balanced`')
lines.append('')
lines.append('Seed candidate:')
lines.append('- hash: `8ba9e6b438e4c909`')
lines.append('- val accuracy: `0.390625`')
lines.append('- solver setting: `temperature=0.6`, seeds `1001, 1002, 1003, 1004`')
lines.append('')
lines.append('Selection rule:')
lines.append(f'- included rows: `{len(regressions)}` validation rows with `mean_solver_accuracy < passrate`')
lines.append('- representative principle-conditioned trace: the first incorrect solver sample if one exists; otherwise the first solver sample')
lines.append('')
lines.append('Source artifacts:')
lines.append('- `candidate_evals/8ba9e6b438e4c909/report.json`')
lines.append('- `candidate_evals/8ba9e6b438e4c909/rows.json`')
lines.append('')
lines.append('Affected row IDs:')
lines.append('- ' + ', '.join(str(row['row_id']) for row in regressions))
lines.append('')

for row in regressions:
    samples = row.get('solver_samples') or []
    representative = next((s for s in samples if not s.get('correct')), samples[0] if samples else None)
    row_id = row['row_id']
    lines.append(f'## Row {row_id}')
    lines.append('')
    lines.append('Metadata:')
    lines.append(f'- row_id: `{row_id}`')
    lines.append(f'- original passrate: `{row["passrate"]}`')
    lines.append(f'- abstraction-conditioned passrate: `{row["mean_solver_accuracy"]}`')
    lines.append(f'- delta: `{float(row["mean_solver_accuracy"]) - float(row["passrate"]):.2f}`')
    lines.append(f'- gold answer: `{row["answer"]}`')
    lines.append(f'- source split: `{row.get("source_split", "")}`')
    if representative is not None:
        lines.append(f'- representative solver seed: `{representative.get("seed")}`')
        lines.append(f'- representative predicted answer: `{representative.get("predicted_answer")}`')
        lines.append(f'- representative correct: `{representative.get("correct")}`')
    lines.append('')
    lines.append('Problem:')
    for problem_line in str(row['problem']).splitlines():
        lines.append(f'> {problem_line}' if problem_line else '>')
    lines.append('')
    lines.append('### Generated principles')
    lines.append('')
    lines.append('```text')
    lines.append((row.get('generated_principles_text') or '').rstrip())
    lines.append('```')
    lines.append('')
    lines.append('### Principle-conditioned trace')
    lines.append('')
    if representative is None:
        lines.append('(No solver samples present.)')
    else:
        lines.append('```text')
        lines.append((representative.get('output_text') or '').rstrip())
        lines.append('```')
    lines.append('')
    lines.append('### Original correct trace')
    lines.append('')
    lines.append('```text')
    lines.append((row.get('first_correct_trace_text') or '').rstrip())
    lines.append('```')
    lines.append('')

out_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
print(out_path)
print(f'rows={len(regressions)}')
