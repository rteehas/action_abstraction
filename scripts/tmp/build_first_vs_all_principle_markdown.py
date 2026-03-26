from __future__ import annotations

import json
from pathlib import Path

ROOT = Path('/workspace/action_abstraction/outputs/2026-03-20/contrastive_abstraction_prompting/principle_v5_first_vs_all_correct_val32_quicktest')
summary = json.loads((ROOT / 'summary.json').read_text())
comparison_rows = json.loads((ROOT / 'comparison_rows.json').read_text())
first_rows = {int(row['row_id']): row for row in json.loads((ROOT / 'first_correct_rows.json').read_text())}
all_rows = {int(row['row_id']): row for row in json.loads((ROOT / 'all_correct_rows.json').read_text())}
comparison_by_id = {int(row['row_id']): row for row in comparison_rows}

helped_ids = [885, 4171, 3300, 3791, 4329]
hurt_ids = [673, 2789, 4457, 2047, 4773]

single_trace_artifact_ids = [2047, 4773]
omitted_helped_single_trace_ids = [3142, 906, 3984]
omitted_hurt_single_trace_ids = [2047, 4773]


def section_for_row(row_id: int) -> str:
    cmp_row = comparison_by_id[row_id]
    first_row = first_rows[row_id]
    all_row = all_rows[row_id]
    problem = first_row['problem'].strip()
    lines = [
        f"## Row {row_id}",
        "",
        f"- original passrate: `{cmp_row['passrate']}`",
        f"- number of non-empty correct traces: `{cmp_row['all_correct_trace_count']}`",
        f"- first-correct score: `{cmp_row['first_correct_score']}`",
        f"- all-correct score: `{cmp_row['all_correct_score']}`",
        f"- delta (`all - first`): `{cmp_row['delta_all_minus_first']}`",
        f"- first correct trace index: `{cmp_row['first_correct_trace_index']}`",
        f"- all correct trace indices: `{cmp_row['all_correct_trace_indices']}`",
    ]
    if row_id in single_trace_artifact_ids:
        lines.extend([
            "- note: this row only has one non-empty correct trace, so the score difference is not attributable to trace-count changes; it reflects rerun stochasticity.",
        ])
    lines.extend([
        "",
        "**Problem**",
        "",
        problem,
        "",
        "**First Correct Only**",
        "",
        "```text",
        (cmp_row['first_principles'] or '').strip(),
        "```",
        "",
        "**All Correct Traces**",
        "",
        "```text",
        (cmp_row['all_principles'] or '').strip(),
        "```",
        "",
    ])
    return "\n".join(lines)

lines = [
    "# Principle Side-by-Side Review",
    "",
    "This note compares `principle_extraction_template_v5.txt` on the same 32-row validation split under two settings:",
    "- first correct trace only",
    "- all non-empty correct traces",
    "",
    "Quick result:",
    f"- first correct trace only val accuracy: `{summary['first_correct']['val_accuracy']}`",
    f"- all correct traces val accuracy: `{summary['all_correct']['val_accuracy']}`",
    f"- delta: `{summary['all_minus_first_val_accuracy']}`",
    f"- rows better with all correct traces: `{summary['rows_better_with_all_correct']}`",
    f"- rows better with first correct trace only: `{summary['rows_better_with_first_correct']}`",
    f"- rows equal: `{summary['rows_equal']}`",
    f"- average number of non-empty correct traces per row: `{summary['avg_all_correct_trace_count']}`",
    "",
    "Selection policy for this note:",
    "- five representative helped rows with more than one non-empty correct trace",
    "- all five hurt rows, with the single-trace rows explicitly marked as stochastic artifacts rather than true first-vs-all comparisons",
    "",
    "Rows omitted from the helped section because they only had one non-empty correct trace:",
    f"- `{omitted_helped_single_trace_ids}`",
    "",
    "## Helped Cases",
    "",
]

for row_id in helped_ids:
    lines.append(section_for_row(row_id))

lines.extend([
    "## Hurt Cases",
    "",
])

for row_id in hurt_ids:
    lines.append(section_for_row(row_id))

out_path = ROOT / 'FIRST_VS_ALL_PRINCIPLES_SIDE_BY_SIDE.md'
out_path.write_text("\n".join(lines), encoding='utf-8')
print(out_path)
