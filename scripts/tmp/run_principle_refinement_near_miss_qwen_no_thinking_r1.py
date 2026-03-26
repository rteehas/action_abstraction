import json
import statistics
import sys
from pathlib import Path

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

REPO = Path('/workspace/action_abstraction')
sys.path.insert(0, str(REPO / 'scripts'))
from contrastive_abstraction_utils import format_solution_block, strip_think_blocks

MODEL = 'Qwen/Qwen3-1.7B'
ROWS_PATH = REPO / 'outputs/2026-03-19/contrastive_abstraction_prompting/combined_principles_v2_v16_passrate_gt0_3k_labeling/rows.json'
PROMPT_PATH = REPO / 'prompt_templates/principle_refinement_template_v1.txt'
OUT_DIR = REPO / 'outputs/2026-03-19/contrastive_abstraction_prompting/principle_refinement_near_miss_qwen_no_thinking_r1'
TARGET_IDXS = [14, 16, 18, 21, 24, 26]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = json.loads(ROWS_PATH.read_text())
    prompt_template = PROMPT_PATH.read_text()

    examples = []
    for idx in TARGET_IDXS:
        row = rows[idx]
        correct_traces = []
        for sol, ok in zip(row.get('generated_solution') or [], row.get('solution_correctness') or []):
            if ok:
                cleaned = strip_think_blocks(sol or '').strip()
                if cleaned:
                    correct_traces.append(cleaned)
        prompt = prompt_template.replace('{{PROBLEM}}', row['problem']).replace(
            '{{CORRECT_SOLUTIONS_BLOCK}}',
            format_solution_block('Correct Trace', correct_traces, max_chars=12000),
        ).replace('{{INITIAL_PRINCIPLES}}', (row.get('principles') or '').strip())
        examples.append({
            'dataset_idx': idx,
            'row_id': row['row_id'],
            'problem': row['problem'],
            'initial_principles': (row.get('principles') or '').strip(),
            'correct_traces': correct_traces,
            'prompt_text': prompt,
        })

    print(f'Loaded {len(examples)} examples')

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    llm = LLM(model=MODEL, max_num_batched_tokens=8192, gpu_memory_utilization=0.9)
    messages = [
        tokenizer.apply_chat_template(
            [{'role': 'user', 'content': ex['prompt_text']}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for ex in examples
    ]
    sampling = SamplingParams(temperature=1.0, top_p=0.95, top_k=20, max_tokens=4096, seed=0)
    outputs = llm.generate(messages, sampling)

    md = []
    md.append('# Principle Refinement Side-By-Side')
    md.append('')
    md.append(f'- Model: `{MODEL}`')
    md.append('- Prompt: `prompt_templates/principle_refinement_template_v1.txt`')
    md.append('- Thinking enabled: `false`')
    md.append(f'- Source rows: `{ROWS_PATH.relative_to(REPO)}`')
    md.append('')

    word_counts_initial = []
    word_counts_refined = []
    trace_counts = []

    for ex, out in zip(examples, outputs):
        refined = out.outputs[0].text.strip()
        ex['refined_principles'] = refined
        ex['raw_output_text'] = refined
        word_counts_initial.append(len(ex['initial_principles'].split()))
        word_counts_refined.append(len(refined.split()))
        trace_counts.append(len(ex['correct_traces']))

        md.append(f"## Dataset Index {ex['dataset_idx']} | Row {ex['row_id']}")
        md.append('')
        md.append(f"- Correct trace count: `{len(ex['correct_traces'])}`")
        md.append('')
        md.append('Problem:')
        md.append('')
        md.append(ex['problem'])
        md.append('')
        md.append('### Original Principles')
        md.append('')
        md.append('```text')
        md.append(ex['initial_principles'])
        md.append('```')
        md.append('')
        md.append('### Refined Principles')
        md.append('')
        md.append('```text')
        md.append(refined)
        md.append('```')
        md.append('')

    report = {
        'model': MODEL,
        'prompt_path': str(PROMPT_PATH.relative_to(REPO)),
        'thinking_enabled': False,
        'num_examples': len(examples),
        'dataset_indices': TARGET_IDXS,
        'mean_correct_trace_count': statistics.mean(trace_counts) if trace_counts else 0.0,
        'mean_initial_words': statistics.mean(word_counts_initial) if word_counts_initial else 0.0,
        'mean_refined_words': statistics.mean(word_counts_refined) if word_counts_refined else 0.0,
        'output_rows_path': 'rows.json',
        'output_markdown_path': 'side_by_side.md',
    }

    (OUT_DIR / 'rows.json').write_text(json.dumps(examples, indent=2))
    (OUT_DIR / 'report.json').write_text(json.dumps(report, indent=2))
    (OUT_DIR / 'side_by_side.md').write_text('\n'.join(md))

    print(json.dumps(report, indent=2))
    print(f'Wrote outputs to {OUT_DIR}')


if __name__ == '__main__':
    main()
