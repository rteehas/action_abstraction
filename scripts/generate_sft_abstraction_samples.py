from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from datasets import Dataset, DatasetDict, load_from_disk
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--abstraction_lora_path", type=str, required=True)
    parser.add_argument("--prompt_template_path", type=str, default="prompt_templates/sft_abstraction_generation.txt")
    parser.add_argument("--source_filter", type=str, default="")
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_num_batched_tokens", type=int, default=4096)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    return parser.parse_args()


def resolve_repo_file(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def load_rows(path: str, source_filter: str, max_rows: int | None) -> list[dict]:
    dataset = load_from_disk(path)
    if isinstance(dataset, DatasetDict):
        if "train" in dataset:
            dataset = dataset["train"]
        else:
            dataset = next(iter(dataset.values()))
    assert isinstance(dataset, Dataset)
    rows = []
    for ex in dataset:
        if source_filter and ex.get("source") != source_filter:
            continue
        rows.append(dict(ex))
        if max_rows is not None and len(rows) >= max_rows:
            break
    return rows


def extract_abstraction(text: str) -> str:
    match = re.search(r"<abstraction>([\s\S]*?)</abstraction>", text or "")
    return match.group(1).strip() if match else (text or "").strip()


def main() -> None:
    args = parse_args()
    rows = load_rows(args.dataset_path, args.source_filter, args.max_rows)
    prompt_template = resolve_repo_file(args.prompt_template_path).read_text(encoding="utf-8")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    prompts = []
    for ex in rows:
        prompt = prompt_template.replace("{{PROBLEM}}", ex["problem"])
        prompts.append(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        )

    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=1.0,
        top_k=-1,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )

    llm = LLM(
        model=args.base_model,
        enable_lora=True,
        max_num_batched_tokens=args.max_num_batched_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    outputs = llm.generate(
        prompts,
        sampling,
        lora_request=LoRARequest("abstraction-generation", 1, args.abstraction_lora_path),
    )

    saved_rows = []
    for idx, (ex, out) in enumerate(zip(rows, outputs)):
        raw_text = out.outputs[0].text
        saved_rows.append(
            {
                "idx": idx,
                "problem": ex["problem"],
                "answer": ex.get("answer"),
                "source": ex.get("source"),
                "reference_abstraction": ex.get("generated_abstraction") or ex.get("abstraction"),
                "sft_output_text": raw_text,
                "sft_abstraction": extract_abstraction(raw_text),
            }
        )

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(saved_rows, indent=2), encoding="utf-8")
    print(f"saved {output_path}")
    print(f"rows={len(saved_rows)}")
    for row in saved_rows[:5]:
        print("IDX", row["idx"])
        print("SFT", row["sft_abstraction"][:500].replace("\n", " "))
        reference = row["reference_abstraction"] or ""
        print("REF", reference[:500].replace("\n", " "))
        print("---")


if __name__ == "__main__":
    main()
