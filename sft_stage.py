from datasets import load_from_disk
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, TaskType
import wandb
from dataclasses import dataclass, field
from typing import List
from argparse import ArgumentParser
import random
from datetime import datetime
import os
from accelerate import Accelerator
from transformers import set_seed
import re

def remove_think_blocks(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)



def get_arguments():
    parser = ArgumentParser()
    parser.add_argument("--with_procedure", action="store_true")
    parser.add_argument("--from_trace", action="store_true")
    parser.add_argument("--dataset_frac", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    return parser


if __name__ == "__main__":
    args = get_arguments().parse_args()
    set_seed(args.seed)
    random.seed(args.seed)
    # dataset = load_from_disk("sft_dataset_from_ppl_model_400")
    dataset = load_from_disk("sft_dataset_from_baseline_no_rl__model")
    
    if args.dataset_frac is not None:
        num_rows = int(args.dataset_frac * dataset.num_rows)

        indices = random.sample(range(dataset.num_rows), k=num_rows)
        dataset = dataset.select(indices)
        print(f"Training with {len(dataset)} rows")

    dataset = dataset.filter(lambda ex: ex["abstraction"] is not None)
    dataset = dataset.filter(lambda ex: ex["abstraction"].strip() != "..." and ex["abstraction"].strip() != "")
    if args.with_procedure:
        dataset = dataset.filter(lambda ex: ex["abstraction"] is not None).filter(lambda ex: ex["procedure"] is not None)

        with open("/scratch/rst306/action_abstractions/action_abstraction/prompt_templates/sft_abstraction_procedure_generation.txt", 'r') as fp:
            lines = fp.readlines()

        sft_prompt_template = "".join(lines)
    
    if args.from_trace:
        dataset = dataset.filter(lambda ex: ex["abstraction"] is not None, num_proc=8)
        with open("/scratch/rst306/action_abstractions/action_abstraction/prompt_templates/sft_problem_reasoning_abstraction_generation_prompt.txt", 'r') as fp:
            lines = fp.readlines()

        sft_prompt_template = "".join(lines)

    else:
        dataset = dataset.filter(lambda ex: ex["abstraction"] is not None, num_proc=8)

        with open("/scratch/rst306/action_abstractions/action_abstraction/prompt_templates/sft_abstraction_generation.txt", 'r') as fp:
            lines = fp.readlines()

        sft_prompt_template = "".join(lines)

    def make_prompt_column(ex):
        problem = ex["problem"]
        prompt = sft_prompt_template.replace("{{PROBLEM}}",problem)
        ex["prompt"] = [{"content": prompt, "role": "user"}]
        return ex

    def make_completion_column(ex):
        abstraction = ex["abstraction"]
        ex["completion"] = [{"content": f"<abstraction>{abstraction}</abstraction>", "role": "assistant"}]
        return ex

    def make_prompt_column_with_trace(ex):
        problem = ex["problem"]
        solution = ex["generated_solution"]
        solution = remove_think_blocks(solution)
        prompt = sft_prompt_template.replace("{{PROBLEM}}", problem).replace("{{REASONING}}", solution)
        ex["prompt"] = [{"content": prompt, "role": "user"}]
        return ex

    with open("/scratch/rst306/action_abstractions/action_abstraction/prompt_templates/abstraction_procedure_block.txt", 'r') as fp:
        lines = fp.readlines()

    abstraction_procedure_block = "".join(lines)

    def make_completion_column_with_procedure(ex):
        abstraction = ex["abstraction"]
        procedure = ex["procedure"]
        completion = abstraction_procedure_block.replace("{{ABSTRACTION}}", abstraction).replace("{{PROCEDURE}}", procedure)
        ex["completion"] = [{"content": completion, "role": "assistant"}]
        return ex

    if args.with_procedure:
        dataset = dataset.map(make_prompt_column).map(make_completion_column_with_procedure)
    if args.from_trace:
        dataset = dataset.map(make_prompt_column_with_trace, num_proc=8).map(make_completion_column, num_proc=8)
    else:
        dataset = dataset.map(make_prompt_column).map(make_completion_column)

    cols_to_keep = ["prompt", "completion"]
    cols_to_drop = [c for c in dataset["train"].column_names if c not in cols_to_keep]
    print(dataset["train"][0])

    dataset = dataset.remove_columns(cols_to_drop)

    # dataset = dataset.train_test_split(test_size=0.1, seed=args.seed)
    print("Dataset columns")
    print(dataset)
    
    acc = Accelerator()
    if acc.is_main_process:
        wandb.init(project="abstraction_generation_sft")
    acc.wait_for_everyone()

    @dataclass
    class LoraArguments:
        lora_r: int = 64
        lora_alpha: int = 16
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
        lora_weight_path: str = ""
        lora_bias: str = "none"
        q_lora: bool = False


    def to_lora_config(args: LoraArguments) -> LoraConfig:
        return LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.lora_target_modules,
            bias=args.lora_bias,          # "none" | "all" | "lora_only"
            task_type=TaskType.CAUSAL_LM, # change if needed
        )

    lora_args = LoraArguments(
        lora_r = 4,
        lora_alpha=8,
    )

    lora_config = to_lora_config(lora_args)

    base_model = "Qwen/Qwen3-1.7B"
    # base_model = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    model_name = base_model.split("/")[1].replace("-", "_").replace(".","_")
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Training {base_model} with model name {model_name}")
    if args.with_procedure:
        output_dir = f"sft_models/{model_name}-abstraction_procedure_generation/{now}"
    if args.from_trace:
        output_dir = f"sft_models/{model_name}-abstraction_generation_from_trace/{now}"
    else:
        output_dir = f"sft_models/{model_name}-abstraction_generation/{now}"

    if args.from_trace:
        batch_size = 8
        gradient_accumulation_steps = 2
    else:
        batch_size = 4
        gradient_accumulation_steps = 4

    training_args = SFTConfig(
        output_dir=output_dir,
        ddp_find_unused_parameters=False,
        max_length=None,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        model_init_kwargs={"device_map": None},              # <- critical for DDP
        per_device_train_batch_size=batch_size,
        adam_beta2=0.95,
        gradient_accumulation_steps=gradient_accumulation_steps,
        dataset_num_proc=8,
        num_train_epochs=6,
        learning_rate=1e-4,
        report_to="wandb",
        save_strategy="epoch",
        eval_strategy="epoch",
        save_total_limit=6,
        seed=args.seed,
        data_seed=args.seed
    )

    trainer = SFTTrainer(
        model = base_model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        peft_config=lora_config,
        args=training_args,
    )

    trainer.train()
