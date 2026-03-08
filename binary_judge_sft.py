from datasets import load_from_disk
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, TaskType
import wandb
from dataclasses import dataclass, field
from typing import List
from argparse import ArgumentParser
import random
import numpy as np
import torch
from transformers import AutoTokenizer
from transformers import TrainerCallback
import torch
from datetime import datetime

class RecordingCollator:
    """Wrap an existing collator and keep the last batch for debugging."""
    def __init__(self, base_collator):
        self.base_collator = base_collator
        self.last_features = None   # list[dict] before tensorization/padding
        self.last_batch = None      # dict[str, torch.Tensor]

    def __call__(self, features):
        self.last_features = features
        batch = self.base_collator(features)
        self.last_batch = batch
        return batch


class BatchInspectCallback(TrainerCallback):
    """
    On each log step:
      - prints current logged loss
      - prints how many non--100 labels each example has
      - prints decoded input and decoded supervised (labeled) tokens for a few examples
    """
    def __init__(
        self,
        collator: RecordingCollator,
        tokenizer,
        max_examples: int = 2,
        max_input_chars: int = 1200,
        every_n_logs: int = 1,
        stop_after: int = 20,
    ):
        self.collator = collator
        self.tokenizer = tokenizer
        self.max_examples = max_examples
        self.max_input_chars = max_input_chars
        self.every_n_logs = every_n_logs
        self.stop_after = stop_after
        self._printed = 0

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or "loss" not in logs:
            return
        if self._printed >= self.stop_after:
            return
        if state.global_step % self.every_n_logs != 0:
            return

        batch = self.collator.last_batch
        if batch is None or "input_ids" not in batch:
            print(f"[step {state.global_step}] loss={logs['loss']:.6f} (no captured batch yet)")
            return

        input_ids = batch["input_ids"]
        labels = batch.get("labels", None)

        # Move to CPU for inspection
        input_ids_cpu = input_ids.detach().cpu()
        labels_cpu = labels.detach().cpu() if labels is not None else None

        bsz, seqlen = input_ids_cpu.shape
        print(f"\n[step {state.global_step}] loss={logs['loss']:.6f}  bsz={bsz} seqlen={seqlen}")

        if labels_cpu is None:
            print("  (no labels in batch)")
            self._printed += 1
            return

        supervised_counts = (labels_cpu != -100).sum(dim=1).tolist()
        print(f"  supervised_tokens_per_example={supervised_counts}")
        if sum(supervised_counts) == 0:
            print("  >>> ALL LABELS ARE -100 IN THIS BATCH (loss will be ~0).")

        n = min(self.max_examples, bsz)
        for i in range(n):
            ids = input_ids_cpu[i].tolist()
            lab = labels_cpu[i].tolist()

            # Full decoded input
            full_text = self.tokenizer.decode(ids, skip_special_tokens=False)
            tail = full_text[-self.max_input_chars:] if len(full_text) > self.max_input_chars else full_text

            # Decoded supervised span
            labeled_pos = [j for j, v in enumerate(lab) if v != -100]
            labeled_ids = [lab[j] for j in labeled_pos]
            labeled_text = self.tokenizer.decode(labeled_ids, skip_special_tokens=False) if labeled_ids else ""

            print(f"\n  --- example {i} ---")
            print(f"  labeled_positions: {labeled_pos[:20]}{' ...' if len(labeled_pos) > 20 else ''}")
            print(f"  labeled_text (decoded): {repr(labeled_text)}")
            print("  input_text_tail:")
            print(tail)

        self._printed += 1


def get_arguments():
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model_name", type=str, default="qwen")
    return parser


if __name__ == "__main__":

    MODEL_NAME_MAP = {
        "qwen": "Qwen/Qwen3-1.7B",
        "deepseek": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    }
    args = get_arguments().parse_args()
    dataset = load_from_disk("grpo_step_200_with_gpt_judgements")
    
    with open("/scratch/rst306/action_abstractions/action_abstraction/prompt_templates/binary_judge_sft_template.txt", 'r') as fp:
        lines = fp.readlines()

    sft_prompt_template = "".join(lines)

    def make_prompt_column(ex):
        problem = ex["problem"]
        abstraction = ex["abstraction"]
        solution_trace = ex["solver_output_text"]
        prompt = sft_prompt_template.replace("{{PROBLEM}}",problem).replace("{{ABSTRACTION}}", abstraction).replace("{{SOLUTION}}", solution_trace)
        ex["prompt"] = [{"content": prompt, "role": "user"}]
        return ex

    def make_completion_column(ex):
        judgement = ex["binary_judgements"].strip()
        ex["completion"] = [{"content": judgement, "role": "assistant"}]
        return ex

    dataset = dataset.map(make_prompt_column).map(make_completion_column)

    cols_to_keep = ["prompt", "completion"]
    cols_to_drop = [c for c in dataset.column_names if c not in cols_to_keep]
    

    dataset = dataset.remove_columns(cols_to_drop)
    print(dataset[0])
    dataset = dataset.train_test_split(test_size=0.1, seed=args.seed)

    wandb.init(project="binary_judge_sft")

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

    base_model = MODEL_NAME_MAP[args.model_name]
    model_name = base_model.split("/")[1].replace("-", "_").replace(".","_")
    print(f"Training {base_model} with model name {model_name}")
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"sft_models/binary_judge/{model_name}/{now}"
    training_args = SFTConfig(
        output_dir=output_dir,
        max_length=None,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        adam_beta2=0.95,
        gradient_accumulation_steps=16,
        # dataset_num_proc=32,
        num_train_epochs=6,
        learning_rate=3e-5,
        report_to="wandb",
        save_strategy="epoch",
        eval_strategy="epoch",
        # eval_steps=2,
        save_total_limit=4,
        seed=args.seed,
        data_seed=args.seed,
        logging_strategy="steps",
        logging_steps=1
    )

    base_model = "Qwen/Qwen3-1.7B"
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    id0 = tokenizer.encode("0", add_special_tokens=False)
    id1 = tokenizer.encode("1", add_special_tokens=False)
    assert len(id0) == 1 and len(id1) == 1, (id0, id1)  # if this fails, use the regex-based variant below
    id0, id1 = id0[0], id1[0]
    print("0", "1")
    print(id0, id1)

    def preprocess_logits_for_metrics(logits, labels):
        # avoid storing full vocab logits for the whole eval set
        if isinstance(logits, tuple):
            logits = logits[0]
        return logits.argmax(dim=-1)

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        preds = np.asarray(preds)
        labels = np.asarray(labels)

        # causal-LM next-token alignment: logits at t predict label at t+1
        preds = preds[:, :-1]
        labels = labels[:, 1:]

        mask = labels != -100
        is_digit = mask & ((labels == id0) | (labels == id1))

        total = correct = 0
        tp = tn = fp = fn = 0

        for i in range(labels.shape[0]):
            pos = np.where(is_digit[i])[0]
            if pos.size == 0:
                continue
            j = int(pos[0])  # only one judgement digit expected
            y = int(labels[i, j] == id1)
            yhat = int(preds[i, j] == id1)

            total += 1
            correct += (y == yhat)

            if y == 1 and yhat == 1: tp += 1
            elif y == 0 and yhat == 0: tn += 1
            elif y == 0 and yhat == 1: fp += 1
            else: fn += 1

        acc = correct / total if total else 0.0
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec  = tp / (tp + fn) if (tp + fn) else 0.0
        f1   = (2 * prec * rec) / (prec + rec) if (prec + rec) else 0.0

        return {
            "judge_acc": acc,
            "judge_f1": f1,
            "judge_tp": tp,
            "judge_tn": tn,
            "judge_fp": fp,
            "judge_fn": fn,
            "judge_n": total,
        }

    trainer = SFTTrainer(
        model = base_model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        peft_config=lora_config,
        args=training_args,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )

    # rec_collator = RecordingCollator(trainer.data_collator)
    # trainer.data_collator = rec_collator
    # trainer.add_callback(BatchInspectCallback(rec_collator, tokenizer, max_examples=2, stop_after=30, max_input_chars=30_000))

    trainer.train()
    # def tok_len(ex):
    #     # whatever TRL is feeding the model is “prompt+completion as chat”
    #     msgs = ex["prompt"] + ex["completion"]
    #     text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    #     ex["n_tokens"] = len(tokenizer(text, add_special_tokens=False).input_ids)
    #     return ex

    # lens = dataset["test"].map(tok_len)
    # train_lens = dataset["train"].map(tok_len)
    # # arr = np.array(lens["n_tokens"])
    # # train_arr = np.array()
    # print("train")
    # print(train_lens["n_tokens"][:10])
    # print("test")
    # print(lens["n_tokens"][:10])
    # print("max", arr.max(), "p99", np.quantile(arr, 0.99), "p95", np.quantile(arr, 0.95))
    # print("top10", sorted(arr)[-10:])
