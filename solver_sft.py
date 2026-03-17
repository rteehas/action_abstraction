from datasets import load_from_disk
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, TaskType
import wandb
from dataclasses import dataclass, field, asdict
from typing import List
from argparse import ArgumentParser
import os
import json
import shutil

from accelerate import Accelerator
from transformers.trainer_utils import get_last_checkpoint


def get_arguments():
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Require an existing checkpoint and resume from it.",
    )
    parser.add_argument(
        "--auto_resume",
        action="store_true",
        help="Resume from the latest checkpoint if one exists; otherwise train from scratch.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete output_dir before training.",
    )
    return parser


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
        bias=args.lora_bias,
        task_type=TaskType.CAUSAL_LM,
    )


def save_run_config(output_dir: str, payload: dict):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "run_config.json"), "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def init_wandb(output_dir: str, project: str):
    wandb_id_file = os.path.join(output_dir, "wandb_run_id.txt")
    os.makedirs(output_dir, exist_ok=True)

    if os.path.exists(wandb_id_file):
        with open(wandb_id_file, "r") as f:
            wandb_id = f.read().strip()
    else:
        wandb_id = wandb.util.generate_id()
        with open(wandb_id_file, "w") as f:
            f.write(wandb_id)

    wandb.init(
        project=project,
        name=os.path.basename(output_dir),
        id=wandb_id,
        resume="allow",
    )


if __name__ == "__main__":
    args = get_arguments().parse_args()

    if args.resume and args.auto_resume:
        raise ValueError("Use only one of --resume or --auto_resume")

    acc = Accelerator()

    base_model = "Qwen/Qwen3-1.7B"
    # base_model = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    model_name = base_model.split("/")[1].replace("-", "_").replace(".", "_")
    output_dir = args.output_dir or f"sft_models/{model_name}-solver"

    if os.path.exists(output_dir) and args.overwrite:
        if acc.is_main_process:
            shutil.rmtree(output_dir)
        acc.wait_for_everyone()

    dataset = load_from_disk("solver_sft_dataset_new")
    # dataset = dataset.filter(lambda ex: ex["abstraction"].strip() != "..." and ex["abstraction"].strip() != "")
    # if args.dataset_frac is not None:
    #     num_rows = int(args.dataset_frac * dataset.num_rows)
    #     indices = random.sample(range(dataset.num_rows), k=num_rows)
    #     dataset = dataset.select(indices)
    #     print(f"Training with {len(dataset)} rows")
    # dataset = dataset.filter(lambda ex: ex["abstraction"] is not None)

    cols_to_keep = ["prompt", "completion"]
    cols_to_drop = [c for c in dataset["train"].column_names if c not in cols_to_keep]

    print(dataset["train"][0])
    dataset["train"] = dataset["train"].remove_columns(cols_to_drop)
    dataset["test"] = dataset["test"].remove_columns(cols_to_drop)

    print("Dataset columns")
    print(dataset)

    last_checkpoint = None
    if os.path.isdir(output_dir):
        found_checkpoint = get_last_checkpoint(output_dir)
        if args.resume:
            if found_checkpoint is None:
                raise ValueError(
                    f"--resume was set, but no checkpoint was found in {output_dir}"
                )
            last_checkpoint = found_checkpoint
        elif args.auto_resume:
            last_checkpoint = found_checkpoint

    if acc.is_main_process:
        print(f"Training {base_model} with model name {model_name}")
        print(f"Output dir: {output_dir}")
        print(f"Last checkpoint: {last_checkpoint}")
        init_wandb(output_dir=output_dir, project="abstraction_solver_sft")

    acc.wait_for_everyone()

    lora_args = LoraArguments(
        lora_r=4,
        lora_alpha=8,
    )
    lora_config = to_lora_config(lora_args)

    batch_size = 2
    gradient_accumulation_steps = 8

    training_args = SFTConfig(
        output_dir=output_dir,
        max_length=None,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        model_init_kwargs={"device_map": None},
        per_device_train_batch_size=batch_size,
        adam_beta2=0.95,
        gradient_accumulation_steps=gradient_accumulation_steps,
        dataset_num_proc=8,
        num_train_epochs=1,
        learning_rate=1e-5,
        report_to="wandb",
        save_total_limit=4,
        save_steps=10,
        save_strategy="steps",
        do_eval=False,
        seed=args.seed,
        data_seed=args.seed,
        logging_steps=1,
    )

    if acc.is_main_process:
        save_run_config(
            output_dir,
            {
                "base_model": base_model,
                "seed": args.seed,
                "batch_size": batch_size,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "lora_args": asdict(lora_args),
            },
        )

    acc.wait_for_everyone()

    trainer = SFTTrainer(
        model=base_model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        peft_config=lora_config,
        args=training_args,
    )

    trainer.train(resume_from_checkpoint=last_checkpoint)
    trainer.save_model()
    trainer.save_state()