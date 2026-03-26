from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from datasets import Dataset, DatasetDict, load_from_disk
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


def extract_abstraction_text(text: str) -> str:
    match = re.search(r"<abstraction>(.*?)</abstraction>", text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    stripped = text.strip()
    stripped = re.sub(r"^<abstraction>\s*", "", stripped, flags=re.IGNORECASE)
    stripped = re.sub(r"\s*</abstraction>$", "", stripped, flags=re.IGNORECASE)
    return stripped.strip()


def load_rows(path: str) -> list[dict]:
    dataset = load_from_disk(path)
    if isinstance(dataset, DatasetDict):
        if 'train' in dataset:
            dataset = dataset['train']
        else:
            dataset = next(iter(dataset.values()))
    assert isinstance(dataset, Dataset)
    return [dict(dataset[i]) for i in range(len(dataset))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--base_model', default='Qwen/Qwen3-1.7B')
    parser.add_argument('--adapter_path', required=True)
    parser.add_argument('--prompt_template_path', required=True)
    parser.add_argument('--temperature', type=float, required=True)
    parser.add_argument('--top_p', type=float, default=0.95)
    parser.add_argument('--top_k', type=int, default=20)
    parser.add_argument('--max_tokens', type=int, default=4096)
    parser.add_argument('--n_abstractions', type=int, default=4)
    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--tensor_parallel_size', type=int, default=1)
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.9)
    parser.add_argument('--max_num_batched_tokens', type=int, default=4096)
    parser.add_argument('--max_rows', type=int, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(args.input_path)
    if args.max_rows is not None:
        rows = rows[: args.max_rows]
    prompt_template = Path(args.prompt_template_path).read_text(encoding='utf-8')
    tokenizer = AutoTokenizer.from_pretrained(args.adapter_path)

    prompts = []
    for row in rows:
        prompt = prompt_template.replace('{{PROBLEM}}', row['problem'])
        chat_text = tokenizer.apply_chat_template(
            [{'role': 'user', 'content': prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prompts.append(chat_text)

    llm = LLM(
        model=args.base_model,
        dtype='bfloat16',
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_batched_tokens=args.max_num_batched_tokens,
        enable_lora=True,
    )
    requested_n = 1 if args.temperature <= 0.0 else args.n_abstractions
    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        n=requested_n,
        seed=args.seed,
    )
    lora_request = LoRARequest('principle_sft_adapter', 1, args.adapter_path)
    outputs = llm.generate(prompts, sampling, use_tqdm=False, lora_request=lora_request)

    flat_rows = []
    for row, output in zip(rows, outputs):
        raw_generations = [candidate.text for candidate in output.outputs]
        if args.temperature <= 0.0 and raw_generations:
            raw_generations = raw_generations * args.n_abstractions
        for abstraction_idx, raw in enumerate(raw_generations[: args.n_abstractions]):
            flat_rows.append({
                **row,
                'abstraction_temperature': args.temperature,
                'abstraction_idx': abstraction_idx,
                'abstraction_raw_generation': raw,
                'generated_principles_text': extract_abstraction_text(raw),
            })

    dataset_path = output_dir / 'dataset'
    Dataset.from_list(flat_rows).save_to_disk(str(dataset_path))
    report = {
        'input_path': args.input_path,
        'output_dataset_path': str(dataset_path),
        'base_model': args.base_model,
        'adapter_path': args.adapter_path,
        'temperature': args.temperature,
        'top_p': args.top_p,
        'top_k': args.top_k,
        'max_tokens': args.max_tokens,
        'n_abstractions': args.n_abstractions,
        'requested_n': requested_n,
        'seed': args.seed,
        'num_input_rows': len(rows),
        'num_output_rows': len(flat_rows),
    }
    (output_dir / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
