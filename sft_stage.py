from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List
import random
import re

from accelerate import Accelerator
from datasets import Dataset, DatasetDict, load_from_disk
from peft import LoraConfig, TaskType
from transformers import set_seed
import wandb


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = "sft_dataset_from_baseline_no_rl__model"
DEFAULT_BASE_MODEL = "Qwen/Qwen3-1.7B"


def repo_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def read_text(path_str: str) -> str:
    return repo_path(path_str).read_text(encoding="utf-8")


def remove_think_blocks(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL)


def word_count(text: str) -> int:
    return len((text or "").split())


def ensure_dataset_dict(dataset: Dataset | DatasetDict, seed: int) -> DatasetDict:
    if isinstance(dataset, DatasetDict):
        return dataset
    if dataset.num_rows <= 1:
        return DatasetDict({"train": dataset, "test": dataset})
    return dataset.train_test_split(test_size=0.1, seed=seed)


def get_arguments() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default=DEFAULT_DATASET)
    parser.add_argument("--prompt_template_path", type=str, default="")
    parser.add_argument("--base_model", type=str, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--output_root", type=str, default="sft_models")
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--with_procedure", action="store_true")
    parser.add_argument("--from_trace", action="store_true")
    parser.add_argument("--dataset_frac", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_train_epochs", type=int, default=6)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--per_device_train_batch_size", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)
    parser.add_argument("--min_abstraction_words", type=int, default=0)
    parser.add_argument("--max_abstraction_words", type=int, default=0)
    parser.add_argument("--save_strategy", type=str, default="epoch")
    parser.add_argument("--eval_strategy", type=str, default="epoch")
    parser.add_argument("--save_steps", type=int, default=0)
    parser.add_argument("--eval_steps", type=int, default=0)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_total_limit", type=int, default=6)
    return parser


def resolve_prompt_template(args) -> str:
    if args.prompt_template_path:
        return read_text(args.prompt_template_path)
    if args.with_procedure:
        return read_text("prompt_templates/sft_abstraction_procedure_generation.txt")
    if args.from_trace:
        return read_text("prompt_templates/sft_problem_reasoning_abstraction_generation_prompt.txt")
    return read_text("prompt_templates/sft_abstraction_generation.txt")


def resolve_output_dir(args) -> str:
    model_name = args.base_model.split("/")[-1].replace("-", "_").replace(".", "_")
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.run_name:
        suffix = args.run_name
    elif args.with_procedure:
        suffix = "abstraction_procedure_generation"
    elif args.from_trace:
        suffix = "abstraction_generation_from_trace"
    else:
        suffix = "abstraction_generation"
    return str(repo_path(args.output_root) / f"{model_name}-{suffix}" / now)


@dataclass
class LoraArguments:
    lora_r: int = 4
    lora_alpha: int = 8
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "up_proj",
            "gate_proj",
            "down_proj",
        ]
    )
    lora_bias: str = "none"


def to_lora_config(args: LoraArguments) -> LoraConfig:
    return LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_target_modules,
        bias=args.lora_bias,
        task_type=TaskType.CAUSAL_LM,
    )


if __name__ == "__main__":
    args = get_arguments().parse_args()
    from trl import SFTConfig, SFTTrainer

    set_seed(args.seed)
    random.seed(args.seed)

    if args.save_strategy not in {"epoch", "steps"}:
        raise ValueError("--save_strategy must be one of: epoch, steps")
    if args.eval_strategy not in {"epoch", "steps"}:
        raise ValueError("--eval_strategy must be one of: epoch, steps")
    if args.save_strategy == "steps" and args.save_steps <= 0:
        raise ValueError("--save_steps must be > 0 when --save_strategy=steps")
    if args.eval_strategy == "steps" and args.eval_steps <= 0:
        raise ValueError("--eval_steps must be > 0 when --eval_strategy=steps")

    dataset = load_from_disk(str(repo_path(args.dataset_path)))
    dataset = ensure_dataset_dict(dataset, seed=args.seed)

    if args.dataset_frac is not None:
        sampled = {}
        for split_name, split_ds in dataset.items():
            if split_ds.num_rows == 0:
                sampled[split_name] = split_ds
                continue
            num_rows = max(1, int(args.dataset_frac * split_ds.num_rows))
            num_rows = min(num_rows, split_ds.num_rows)
            indices = random.sample(range(split_ds.num_rows), k=num_rows)
            sampled[split_name] = split_ds.select(indices)
            print(f"{split_name}: training with {len(sampled[split_name])} rows")
        dataset = DatasetDict(sampled)

    dataset = dataset.filter(lambda ex: ex.get("abstraction") is not None)
    dataset = dataset.filter(lambda ex: ex["abstraction"].strip() not in {"", "..."})
    if args.min_abstraction_words > 0:
        dataset = dataset.filter(lambda ex: word_count(ex["abstraction"]) >= args.min_abstraction_words)
    if args.max_abstraction_words > 0:
        dataset = dataset.filter(lambda ex: word_count(ex["abstraction"]) <= args.max_abstraction_words)

    if args.with_procedure:
        dataset = dataset.filter(lambda ex: ex.get("procedure") is not None)

    sft_prompt_template = resolve_prompt_template(args)
    abstraction_procedure_block = read_text("prompt_templates/abstraction_procedure_block.txt")

    def make_prompt_column(ex):
        prompt = sft_prompt_template.replace("{{PROBLEM}}", ex["problem"])
        ex["prompt"] = [{"content": prompt, "role": "user"}]
        return ex

    def make_prompt_column_with_trace(ex):
        solution = remove_think_blocks(ex.get("generated_solution", ""))
        prompt = sft_prompt_template.replace("{{PROBLEM}}", ex["problem"]).replace("{{REASONING}}", solution)
        ex["prompt"] = [{"content": prompt, "role": "user"}]
        return ex

    def make_completion_column(ex):
        ex["completion"] = [{"content": f"<abstraction>{ex['abstraction']}</abstraction>", "role": "assistant"}]
        return ex

    def make_completion_column_with_procedure(ex):
        completion = abstraction_procedure_block.replace("{{ABSTRACTION}}", ex["abstraction"]).replace("{{PROCEDURE}}", ex["procedure"])
        ex["completion"] = [{"content": completion, "role": "assistant"}]
        return ex

    if args.with_procedure:
        dataset = dataset.map(make_prompt_column, num_proc=8).map(make_completion_column_with_procedure, num_proc=8)
    elif args.from_trace:
        dataset = dataset.map(make_prompt_column_with_trace, num_proc=8).map(make_completion_column, num_proc=8)
    else:
        dataset = dataset.map(make_prompt_column, num_proc=8).map(make_completion_column, num_proc=8)

    cols_to_keep = {"prompt", "completion"}
    cols_to_drop = [c for c in dataset["train"].column_names if c not in cols_to_keep]
    print(dataset["train"][0])
    dataset = dataset.remove_columns(cols_to_drop)

    print("Dataset columns")
    print(dataset)

    acc = Accelerator()
    if acc.is_main_process:
        wandb.init(project="abstraction_generation_sft")
    acc.wait_for_everyone()

    lora_config = to_lora_config(LoraArguments())

    if args.from_trace:
        batch_size = args.per_device_train_batch_size or 8
        grad_accum = args.gradient_accumulation_steps or 2
    else:
        batch_size = args.per_device_train_batch_size or 4
        grad_accum = args.gradient_accumulation_steps or 4

    training_args = SFTConfig(
        output_dir=resolve_output_dir(args),
        ddp_find_unused_parameters=False,
        max_length=None,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        model_init_kwargs={"device_map": None},
        per_device_train_batch_size=batch_size,
        adam_beta2=0.95,
        gradient_accumulation_steps=grad_accum,
        dataset_num_proc=8,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        report_to="wandb",
        save_strategy=args.save_strategy,
        eval_strategy=args.eval_strategy,
        save_steps=args.save_steps if args.save_strategy == "steps" else None,
        eval_steps=args.eval_steps if args.eval_strategy == "steps" else None,
        logging_steps=args.logging_steps,
        save_total_limit=args.save_total_limit,
        seed=args.seed,
        data_seed=args.seed,
    )

    trainer = SFTTrainer(
        model=args.base_model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        peft_config=lora_config,
        args=training_args,
    )

    print(f"Training {args.base_model}")
    print(f"Output dir: {training_args.output_dir}")
    trainer.train()
