import json
from pathlib import Path
import sys

from datasets import load_from_disk
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

REPO = Path('/workspace/action_abstraction')
sys.path.insert(0, str(REPO))

from eval_abstraction_conditioned_baseline import apply_chat_template, cleanup_model, render_template, score_answer
from process_deepscaler_dataset import extract_answer_from_solution

PROMPT_PATH = REPO / 'prompt_templates' / 'hint_conditioned_problem_solving_rich_v2txt'
SOURCE_DATASET = REPO / 'outputs' / '2026-03-25' / 'contrastive_abstraction_prompting' / 'hierarchical_principle_baseline_qwen3_1_7b_4x4_rich_v1' / 'scored_dataset'
OUTPUT_DIR = REPO / 'outputs' / '2026-03-25' / 'contrastive_abstraction_prompting' / 'solver_spotcheck_rich_v2txt_rows_10_12_24_29'
ROW_IDS = [10, 12, 24, 29]
SOLVER_MODEL = 'Qwen/Qwen3-1.7B'
TEMPERATURE = 0.6
TOP_P = 0.95
TOP_K = 20
MAX_TOKENS = 8192
SEED = 1234


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset = load_from_disk(str(SOURCE_DATASET))
    prompt_template = PROMPT_PATH.read_text(encoding='utf-8')
    selected_rows = [dataset[i] for i in ROW_IDS]

    tokenizer = AutoTokenizer.from_pretrained(SOLVER_MODEL)
    raw_prompts = []
    chat_prompts = []
    prompt_index = []

    for row in selected_rows:
        for abstraction_idx, abstraction in enumerate(row['generated_abstractions']):
            raw_prompt = render_template(
                prompt_template,
                problem=row['problem'],
                abstraction=abstraction,
            )
            raw_prompts.append(raw_prompt)
            chat_prompt = apply_chat_template(
                tokenizer,
                raw_prompt,
                enable_thinking=True,
            )
            chat_prompts.append(chat_prompt)
            prompt_index.append((int(row['row_idx']), abstraction_idx))

    llm = LLM(
        model=SOLVER_MODEL,
        dtype='bfloat16',
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        max_num_batched_tokens=4096,
    )
    sampling_params = SamplingParams(
        temperature=TEMPERATURE,
        top_p=TOP_P,
        top_k=TOP_K,
        max_tokens=MAX_TOKENS,
        n=1,
        seed=SEED,
    )
    outputs = llm.generate(chat_prompts, sampling_params)
    cleanup_model(llm)

    rows_by_id = {int(row['row_idx']): row for row in selected_rows}
    full_results = []
    summary = []

    for prompt_idx, output in enumerate(outputs):
        row_idx, abstraction_idx = prompt_index[prompt_idx]
        row = rows_by_id[row_idx]
        solution = output.outputs[0].text
        extracted_answer = extract_answer_from_solution(solution)
        correct = bool(score_answer(row['answer'], extracted_answer))
        full_results.append(
            {
                'row_idx': row_idx,
                'source': row['source'],
                'answer': row['answer'],
                'abstraction_idx': abstraction_idx,
                'extracted_answer': extracted_answer,
                'correct': correct,
                'problem': row['problem'],
                'generated_abstraction': row['generated_abstractions'][abstraction_idx],
                'raw_prompt': raw_prompts[prompt_idx],
                'generated_solution': solution,
                'baseline_answers_for_same_abstraction': row['generated_answer_by_abstraction'][abstraction_idx],
                'baseline_num_correct_for_same_abstraction': int(row['num_correct_by_abstraction'][abstraction_idx]),
            }
        )

    for row_idx in ROW_IDS:
        row_records = [record for record in full_results if record['row_idx'] == row_idx]
        summary.append(
            {
                'row_idx': row_idx,
                'source': row_records[0]['source'],
                'answer': row_records[0]['answer'],
                'num_correct_out_of_4': sum(1 for record in row_records if record['correct']),
                'extracted_answers': [record['extracted_answer'] for record in row_records],
                'correct_flags': [record['correct'] for record in row_records],
            }
        )

    report = {
        'prompt_path': str(PROMPT_PATH),
        'solver_model': SOLVER_MODEL,
        'row_ids': ROW_IDS,
        'temperature': TEMPERATURE,
        'top_p': TOP_P,
        'top_k': TOP_K,
        'max_tokens': MAX_TOKENS,
        'num_solver_samples_per_abstraction': 1,
        'used_saved_abstractions_from': str(SOURCE_DATASET),
        'summary': summary,
    }

    (OUTPUT_DIR / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    (OUTPUT_DIR / 'full_results.json').write_text(json.dumps(full_results, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
