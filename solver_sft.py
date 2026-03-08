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


def get_arguments():
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    return parser


if __name__ == "__main__":
    args = get_arguments().parse_args()
    dataset = load_from_disk("solver_sft_dataset_partial")
    # dataset = dataset.filter(lambda ex: ex["abstraction"].strip() != "..." and ex["abstraction"].strip() != "")
    # if args.dataset_frac is not None:
    #     num_rows = int(args.dataset_frac * dataset.num_rows)

    #     indices = random.sample(range(dataset.num_rows), k=num_rows)
    #     dataset = dataset.select(indices)
    #     print(f"Training with {len(dataset)} rows")
    
    # dataset = dataset.filter(lambda ex: ex["abstraction"] is not None)

    cols_to_keep = ["prompt", "completion"]
    cols_to_drop = [c for c in dataset.column_names if c not in cols_to_keep]
    print(dataset[0])

    dataset = dataset.remove_columns(cols_to_drop)

    # dataset = dataset.train_test_split(test_size=0.1, seed=args.seed)
    print("Dataset columns")
    print(dataset)
    
    acc = Accelerator()
    if acc.is_main_process:
        wandb.init(project="abstraction_solver_sft")
        
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
    output_dir = f"sft_models/{model_name}-solver/{now}"

    batch_size = 2
    gradient_accumulation_steps= 8

    training_args = SFTConfig(
        output_dir=output_dir,
        max_length=None,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        model_init_kwargs={"device_map": None},              # <- critical for DDP
        per_device_train_batch_size=batch_size,
        adam_beta2=0.95,
        gradient_accumulation_steps=gradient_accumulation_steps,
        dataset_num_proc=8,
        num_train_epochs=1,
        learning_rate=1e-4,
        report_to="wandb",
        save_total_limit=4,
        save_steps=10,
        do_eval=False,
        seed=args.seed,
        data_seed=args.seed
    )

    trainer = SFTTrainer(
        model = base_model,
        train_dataset=dataset,
        # eval_dataset=dataset["test"],
        peft_config=lora_config,
        args=training_args,
    )

    trainer.train()
