from __future__ import annotations

import argparse
import sys
from pathlib import Path

from datasets import Dataset, DatasetDict, load_from_disk
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from contrastive_abstraction_utils import extract_abstraction, read_text, repo_path
from eval_contrastive_abstractions import render_label_prompt


DEFAULT_INPUT = 'contrastive_abstraction_datasets/deepscaler_mixed_correct_incorrect_256'
DEFAULT_OUTPUT = 'sft_datasets/contrastive_abstraction_labels_v1'
DEFAULT_PROMPT = 'prompt_templates/contrastive_abstraction_labeling_prompt_v2.txt'
DEFAULT_FEW_SHOTS = 'prompt_templates/contrastive_abstraction_labeling_few_shots.txt'
DEFAULT_BASE_MODEL = 'Qwen/Qwen3-1.7B'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dataset_path', type=str, default=DEFAULT_INPUT)
    parser.add_argument('--output_dataset_path', type=str, default=DEFAULT_OUTPUT)
    parser.add_argument('--split', type=str, default='', help='Optional single split to process; default processes all splits.')
    parser.add_argument('--max_rows', type=int, default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--base_model', type=str, default=DEFAULT_BASE_MODEL)
    parser.add_argument('--abstraction_lora_path', type=str, default='')
    parser.add_argument('--prompt_path', type=str, default=DEFAULT_PROMPT)
    parser.add_argument('--few_shots_path', type=str, default=DEFAULT_FEW_SHOTS)
    parser.add_argument('--label_mode', choices=['contrastive', 'problem_only'], default='contrastive')
    parser.add_argument('--max_solution_chars', type=int, default=6000)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--max_tokens', type=int, default=4096)
    parser.add_argument('--max_model_len', type=int, default=None)
    parser.add_argument('--max_num_batched_tokens', type=int, default=8192)
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.9)
    parser.add_argument('--filter_empty', action='store_true')
    return parser.parse_args()


def resolve_repo_file(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return repo_path(path_str)


def load_dataset(path: str) -> DatasetDict:
    ds = load_from_disk(str(resolve_repo_file(path)))
    if isinstance(ds, DatasetDict):
        return ds
    return DatasetDict({'train': ds})


def maybe_limit(ds: Dataset, max_rows: int | None) -> Dataset:
    if max_rows is None:
        return ds
    return ds.select(range(min(max_rows, ds.num_rows)))


def generate_with_optional_lora(llm: LLM, prompts: list[str], sampling: SamplingParams, lora_path: str) -> list[str]:
    if lora_path:
        outputs = llm.generate(prompts, sampling, lora_request=LoRARequest('abstraction-generation', 1, lora_path))
    else:
        outputs = llm.generate(prompts, sampling)
    return [output.outputs[0].text for output in outputs]


def build_rows(split_ds: Dataset, tokenizer: AutoTokenizer, template: str, few_shots: str, args: argparse.Namespace, llm: LLM) -> list[dict]:
    prompts = [
        render_label_prompt(ex, template, few_shots, args.label_mode, args.max_solution_chars)
        for ex in split_ds
    ]
    messages = [
        tokenizer.apply_chat_template(
            [{'role': 'user', 'content': prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for prompt in prompts
    ]
    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=0.95,
        top_k=20,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )
    outputs = generate_with_optional_lora(llm, messages, sampling, args.abstraction_lora_path)

    rows = []
    for ex, prompt_text, output_text in zip(split_ds, prompts, outputs):
        abstraction = extract_abstraction(output_text)
        row = dict(ex)
        row['label_prompt_text'] = prompt_text
        row['generated_abstraction_text'] = output_text
        row['abstraction'] = abstraction
        rows.append(row)
    if args.filter_empty:
        rows = [row for row in rows if row.get('abstraction')]
    return rows


def main() -> None:
    args = parse_args()
    dataset = load_dataset(args.input_dataset_path)
    if args.split:
        dataset = DatasetDict({args.split: dataset[args.split]})

    template = read_text(resolve_repo_file(args.prompt_path))
    few_shots = read_text(resolve_repo_file(args.few_shots_path)) if args.few_shots_path else ''
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    llm_kwargs = {
        'model': args.base_model,
        'enable_lora': bool(args.abstraction_lora_path),
        'max_num_batched_tokens': args.max_num_batched_tokens,
        'gpu_memory_utilization': args.gpu_memory_utilization,
    }
    if args.max_model_len is not None:
        llm_kwargs['max_model_len'] = args.max_model_len
    llm = LLM(**llm_kwargs)

    built_splits = {}
    for split_name, split_ds in dataset.items():
        split_ds = maybe_limit(split_ds, args.max_rows)
        rows = build_rows(split_ds, tokenizer, template, few_shots, args, llm)
        built_splits[split_name] = Dataset.from_list(rows)
        print(f'{split_name}: kept {built_splits[split_name].num_rows} rows')
        if built_splits[split_name].num_rows:
            print(built_splits[split_name][0]['abstraction'])

    output_path = resolve_repo_file(args.output_dataset_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    DatasetDict(built_splits).save_to_disk(str(output_path))
    print(f'Saved labeled SFT dataset to {output_path}')


if __name__ == '__main__':
    main()
