import json
from pathlib import Path

REPO_ROOT = Path('/workspace/action_abstraction')
OUT_DIR = REPO_ROOT / 'outputs/2026-03-19/contrastive_abstraction_prompting/principle_extractions_tracewise_qwen_v3_r1'
rows = json.loads((OUT_DIR / 'rows.json').read_text())
report = json.loads((OUT_DIR / 'report.json').read_text())

order = []
by_row = {}
for r in rows:
    rid = r['row_id']
    if rid not in by_row:
        order.append(rid)
        by_row[rid] = {'problem': r['problem'], 'items': []}
    by_row[rid]['items'].append(r)

lines = [
    '# Principle Extractions By Problem',
    '',
    f"- model: `{report['model_name']}`",
    '- prompt: `prompt_templates/principle_extraction_template_v3.txt`',
    '- problems: 26',
    f"- trace prompts: {report['num_trace_prompts']}",
    '',
]

for rid in order:
    entry = by_row[rid]
    lines.append(f'## Row {rid}')
    lines.append('')
    lines.append('**Problem**')
    lines.append(entry['problem'].strip())
    lines.append('')
    for extracted in entry['items']:
        lines.append(f"### Correct Trace {extracted['trace_local_idx']}")
        lines.append('')
        lines.append('**Trace**')
        lines.append('```text')
        lines.append(extracted['trace_text'].rstrip())
        lines.append('```')
        lines.append('')
        lines.append('**Extracted Principles**')
        lines.append('```text')
        lines.append(extracted['generated_principles'].rstrip())
        lines.append('```')
        lines.append('')
    lines.append('---')
    lines.append('')

path = OUT_DIR / 'principles_by_problem.md'
path.write_text('\n'.join(lines), encoding='utf-8')
print(path)
