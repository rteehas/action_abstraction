from datasets import load_from_disk
from trl import GRPOTrainer, GRPOConfig
# from problem_solving import make_problem_solving_prompt
from abstraction_generation_processing import extract_abstraction_from_tag
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from functools import partial
import re
import wandb
from datasets import Dataset
from transformers import TrainerCallback
import threading, shutil, os
from math_verify import parse, verify
from dataclasses import dataclass, field
from peft import LoraConfig, TaskType
from typing import List
from argparse import ArgumentParser
from datetime import datetime
import time
import random
from tqdm import tqdm
import re
from dataclasses import asdict, is_dataclass
from pathlib import Path
import json
import torch.nn.functional as F


def get_distributed_context():
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return local_rank, rank, world_size


def get_process_device(local_rank: int) -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")
    if local_rank >= 0:
        torch.cuda.set_device(local_rank)
        return torch.device(f"cuda:{local_rank}")
    return torch.device("cuda:0")


def remove_think_blocks(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

def extract_think(text):
    pattern = r"<think>.*?</think>"
    m = re.search(pattern, text, flags=re.DOTALL)
    return m.group(0) if m else None

# top-level init (once)
with open("/scratch/rst306/action_abstractions/action_abstraction/prompt_templates/hint_conditioned_problem_solving.txt", "r") as f:
    PROBLEM_SOLVING_TMPL = f.read()

def make_problem_solving_prompt(problem, abstraction):
    # # with open("/scratch/rst306/action_abstractions/action_abstraction/prompt_templates/hint_conditioned_problem_solving.txt", 'r') as fp:
    # #     lines = fp.readlines()
    
    # problem_solving_prompt_template = "".join(lines)

    prompt = PROBLEM_SOLVING_TMPL.replace("{{PROBLEM}}", problem).replace("{{ABSTRACTION}}", abstraction)
    return prompt


def extract_answer_from_solution(text):
    m = list(re.finditer(r'\\boxed\s*\{', text))
    if not m:
        return None

    box_start = m[-1].start()   # include "\boxed{"
    start = m[-1].end()         # position right after "{"

    depth = 1
    i = start
    while i < len(text) and depth > 0:
        c = text[i]
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
        i += 1

    if depth != 0:
        return None

    return text[box_start:i]    # includes "\boxed{...}"

# class RolloutLogger:
#     """
#     Accumulates per-completion rollout records and writes them as a HF Dataset.
#     Saves:
#       - {root}/latest              (overwritten each save)
#       - {root}/step-{global_step}  (snapshot per checkpoint)
#     """
#     def __init__(self, root_dir: str):
#         self.root_dir = root_dir
#         os.makedirs(self.root_dir, exist_ok=True)
#         self._lock = threading.Lock()
#         self._rows = []
#         self._step = 0

#     def set_step(self, step: int):
#         self._step = int(step)

#     def add_rows(self, rows):
#         with self._lock:
#             # stamp step at insertion time
#             for r in rows:
#                 r.setdefault("global_step", self._step)
#             self._rows.extend(rows)

#     def _save_dir(self, path: str):
#         if os.path.exists(path):
#             shutil.rmtree(path)
#         ds = Dataset.from_list(self._rows) if self._rows else Dataset.from_list([])
#         ds.save_to_disk(path)

#     def save(self, step: int):
#         self.set_step(step)
#         latest_dir = os.path.join(self.root_dir, "latest")
#         snap_dir = os.path.join(self.root_dir, f"step-{int(step)}")
#         with self._lock:
#             self._save_dir(latest_dir)
#             # self._save_dir(snap_dir)

from datasets import load_from_disk, Dataset
import os, shutil, threading

class RolloutLogger:
    """
    Accumulates per-completion rollout records and writes them as a HF Dataset.
    On resume, loads existing rows from {root}/latest and keeps appending.
    Saves:
      - {root}/latest              (overwritten each save, but contains ALL rows so far)
      - {root}/step-{global_step}  (optional snapshot)
    """
    def __init__(self, root_dir: str, load_existing: bool = True):
        self.root_dir = root_dir
        os.makedirs(self.root_dir, exist_ok=True)
        self._lock = threading.Lock()
        self._rows = []
        self._step = 0

        if load_existing:
            self._load_existing_latest()

    def _load_existing_latest(self):
        latest_dir = os.path.join(self.root_dir, "latest")
        if not os.path.exists(latest_dir):
            return
        try:
            ds = load_from_disk(latest_dir)
            rows = ds.to_list()
            with self._lock:
                self._rows.extend(rows)
        except Exception:
            # if corrupted/partial, just skip loading
            return

    def set_step(self, step: int):
        self._step = int(step)

    def add_rows(self, rows):
        with self._lock:
            for r in rows:
                r.setdefault("global_step", self._step)
            self._rows.extend(rows)

    def _save_dir(self, path: str):
        if os.path.exists(path):
            shutil.rmtree(path)
        ds = Dataset.from_list(self._rows) if self._rows else Dataset.from_list([])
        ds.save_to_disk(path)

    def save(self, step: int):
        self.set_step(step)
        latest_dir = os.path.join(self.root_dir, "latest")
        # snap_dir = os.path.join(self.root_dir, f"step-{int(step)}")
        with self._lock:
            self._save_dir(latest_dir)
            # self._save_dir(snap_dir)


class RolloutSaveCallback(TrainerCallback):
    def __init__(self, logger: RolloutLogger):
        self.logger = logger

    def on_step_end(self, args, state, control, **kwargs):
        self.logger.set_step(state.global_step)

    def on_save(self, args, state, control, **kwargs):
        self.logger.save(state.global_step)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        self.logger.save(state.global_step)
        return control


def get_few_shot_examples():
    dataset = load_from_disk("/vast/rst306/action_abstraction/open_math_reasoning_104K_extracted_filtered")
    dataset = dataset.filter(lambda ex: ex["abstraction"] is not None and ex["abstraction"] != "")
    return dataset.select(list(range(3))) # 3 shot

def make_few_shot_example(problem, abstraction, template):
    return template.replace("{{PROBLEM}}", problem).replace("{{ABSTRACTION_LABEL}}", abstraction)

def make_few_shot_block(few_shot_dataset):
    with open("/scratch/rst306/action_abstractions/action_abstraction/prompt_templates/abstraction_generation_few_shot_example.txt", 'r') as fp:
        lines = fp.readlines()
    
    few_shot_example_template = "".join(lines)
    few_shot_examples = [make_few_shot_example(ex["problem"], ex["abstraction"], few_shot_example_template) for ex in few_shot_dataset]
    return "\n\n".join(few_shot_examples)

# @torch.inference_mode()
# def compute_ppl_batched(prefix_lens, texts, tokenizer, model):
#     enc = tokenizer(
#         texts,
#         add_special_tokens=False,
#         padding=True,
#         return_tensors="pt",
#     )
#     input_ids = enc["input_ids"].to(model.device, non_blocking=True)
#     attn = enc["attention_mask"].to(model.device, non_blocking=True)

#     B, T = input_ids.shape

#     # build labels = input_ids but mask prefix tokens
#     labels = input_ids.clone()
#     ar = torch.arange(T, device=labels.device).unsqueeze(0)  # [1,T]
#     pl = torch.tensor(prefix_lens, device=labels.device).unsqueeze(1)  # [B,1]
#     labels[ar < pl] = -100

#     # forward
#     out = model(input_ids=input_ids, attention_mask=attn, use_cache=False)
#     logits = out.logits  # [B,T,V]

#     # next-token prediction
#     shift_logits = logits[:, :-1, :].contiguous()
#     shift_labels = labels[:, 1:].contiguous()
#     shift_attn = attn[:, 1:].contiguous()

#     V = shift_logits.size(-1)
#     loss_tok = F.cross_entropy(
#         shift_logits.view(-1, V),
#         shift_labels.view(-1),
#         reduction="none",
#         ignore_index=-100,
#     ).view(B, T - 1)

#     valid = (shift_labels != -100) & (shift_attn == 1)
#     nll_sum = (loss_tok * valid).sum(dim=1)
#     n_valid = valid.sum(dim=1).clamp_min(1)

#     ppl = torch.exp((nll_sum / n_valid).to(torch.float32)).cpu().tolist()
#     return ppl

def compute_ppl_batched(prompt_lens, texts, tokenizer, model):
    
    enc = tokenizer(
        texts,
        add_special_tokens=False,
        padding=True,
        return_tensors="pt",
    )

    input_ids = enc["input_ids"].to(model.device)
    attn = enc["attention_mask"].to(model.device)

    labels = input_ids.clone()

    # mask out the prompt tokens so loss is only on solution tokens
    for i, pl in enumerate(prompt_lens):
        # if truncated, pl might exceed seq len
        pl = min(pl, labels.size(1))
        labels[i, :pl] = -100

    out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
    # HF loss is mean NLL over non -100 labels in the batch
    # Convert to per-example perplexities by computing token-level NLL per row.
    logits = out.logits  # [B, T, V]

    # shift for next-token prediction
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    shift_attn = attn[:, 1:].contiguous()

    log_probs = torch.log_softmax(shift_logits, dim=-1)
    # gather logp of the gold token ids (ignore -100 later)
    gold = shift_labels.clamp_min(0).unsqueeze(-1)  # avoid gather crash
    tok_logp = log_probs.gather(-1, gold).squeeze(-1)  # [B, T-1]

    valid = (shift_labels != -100) & (shift_attn == 1)
    # sum NLL per example / number of valid tokens per example
    nll_sum = (-tok_logp * valid).sum(dim=1)
    n_valid = valid.sum(dim=1).clamp_min(1)
    nll_mean = (nll_sum / n_valid).to(torch.float32)
    # print(nll_sum, n_valid, input_ids.shape)
    ppl = torch.exp(nll_mean).cpu().tolist()
    return ppl

# @torch.no_grad()
# def compute_ppl_batched(prompt_lens, texts, tokenizer, model):
#     enc = tokenizer(
#         texts,
#         add_special_tokens=False,
#         padding=True,
#         return_tensors="pt",
#     )

#     input_ids = enc["input_ids"].to(model.device)
#     attn = enc["attention_mask"].to(model.device)

#     labels = input_ids.clone()

#     # mask out the prompt tokens so loss is only on solution tokens
#     for i, pl in enumerate(prompt_lens):
#         pl = min(pl, labels.size(1))
#         labels[i, :pl] = -100

#     with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
#         logits = model(input_ids=input_ids, attention_mask=attn, use_cache=False).logits

#     # shift for next-token prediction
#     shift_logits = logits[:, :-1, :].contiguous()
#     shift_labels = labels[:, 1:].contiguous()
#     shift_attn = attn[:, 1:].contiguous()

#     B, T = shift_logits.shape[:2]
#     V = shift_logits.shape[2]

#     # Fused cross_entropy: avoids materializing a full [B, T, V] log_softmax
#     # tensor.  F.cross_entropy computes log_softmax + NLL internally in a
#     # single fused kernel, so peak memory drops by ~sizeof(float)*B*T*V.
#     loss_tok = F.cross_entropy(
#         shift_logits.view(-1, V),
#         shift_labels.view(-1),
#         reduction="none",
#         ignore_index=-100,
#     ).view(B, T)

#     valid = (shift_labels != -100) & (shift_attn == 1)
#     nll_sum = (loss_tok * valid).sum(dim=1)
#     n_valid = valid.sum(dim=1).clamp_min(1)
#     nll_mean = (nll_sum / n_valid).to(torch.float32)

#     ppl = torch.exp(nll_mean).cpu().tolist()
#     return ppl



@torch.no_grad()
def ppl_reward_function(prompts, completions, completion_ids, problem, correct_solution, generated_solution_ppl, unconditional_solution_ppl, model, tokenizer, batch_size, no_think=False, clip_reward=False, rollout_logger=None, subtract_compression_ppl=False, unconditional_lambda=0, **kwargs):
    n = len(completions)
    reward = [0.0] * n
    abstractions = [None] * n
    solving_prompts = [None] * n
    prefixes = [None] * n
    

    prefix_lens = []
    full_prompts = []

    idxs = []
    model_ppls = []
    model_ppls_for_recording = [None] * n
    full_prompts_for_recording = [None] * n
    compression_ppl_lens = []
    baseline_compression_ppl_lens = []
    full_prompts_compression = []
    full_prompts_compression_baseline = []

    full_prompts_unconditional = []
    unconditional_prefixes = []
    unconditional_prefix_lens = []
    unconditional_ppls = []

    unconditional_ppls_for_recording = [None] * n 
    full_prompts_unconditional_for_recording = [None] * n
    unconditional_prompts_for_recording = [None] * n

    ppl_improvements = []
    unconditional_improvements = []
    ppl_improvement_for_recording = [None] * n
    unconditional_improvement_for_recording = [None] * n
    print("all prompts", len(prompts))
    print("all distinct prompts", len(list(set(prompts))))
    # print("all completions", n)
    # for p in prompts:
    #     print(p)
    for i, strategy in enumerate(completions):
        abstraction = extract_abstraction_from_tag(strategy)
        abstractions[i] = abstraction
        if abstraction is None or abstraction.strip() == "" or abstraction.strip() == "..." or "\\boxed" in abstraction: #last one is a hack, not sure why it's generating those, check 
            reward[i] = 0.0
            continue

        prob = problem[i]
        correct_sol = correct_solution[i]

        solving_prompt = make_problem_solving_prompt(prob, abstraction.strip())
        solving_prompts[i] = solving_prompt
        messages = [{"role": "user", "content": solving_prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        ).replace("<think>", "").replace("</think>", "").strip()

        if no_think:
            think_block = extract_think(correct_sol)
            remainder = remove_think_blocks(correct_sol)
            text = text + "\n" + think_block

            pl = tokenizer(text, add_special_tokens=False).input_ids
            prefix_lens.append(len(pl))
            full_prompt = text + remainder
        else:
            pl = tokenizer(text, add_special_tokens=False).input_ids
            prefix_lens.append(len(pl))

            full_prompt = text + "\n" + correct_sol.strip()
        
        unconditional_prompt = abstraction.strip()
        messages = [{"role": "user", "content": unconditional_prompt}]
        unconditional_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        unconditional_pl = tokenizer(unconditional_text, add_special_tokens=False).input_ids
        unconditional_prefix_lens.append(len(unconditional_pl))

        think_block = extract_think(correct_sol)
        remainder = remove_think_blocks(correct_sol)
        full_unconditional_prompt = unconditional_text + remainder
        full_prompts_unconditional.append(full_unconditional_prompt)
        unconditional_prefixes.append(unconditional_text)
        # if subtract_compression_ppl:
        unconditional_prompts_for_recording[i] = full_unconditional_prompt


        # print("decoded prefix")
        # enc_text = tokenizer(full_prompt, add_special_tokens=False).input_ids
        # print(tokenizer.decode(enc_text[:len(pl)]))
        prefixes[i] = text
        full_prompts.append(full_prompt)
        full_prompts_for_recording[i] = full_prompt
        idxs.append(i)
    
    for i in range(0, len(full_prompts), batch_size):
        lens = prefix_lens[i: i + batch_size]
        full_ps = full_prompts[i: i + batch_size]
        ppls = compute_ppl_batched(lens, full_ps, tokenizer, model)
        model_ppls += ppls
        # torch.cuda.empty_cache()
    
    for i in range(0, len(full_prompts_unconditional), batch_size):
        lens = unconditional_prefix_lens[i: i + batch_size]
        full_ps = full_prompts_unconditional[i: i + batch_size]
        ppls = compute_ppl_batched(lens, full_ps, tokenizer, model)
        unconditional_ppls += ppls
    
    # torch.cuda.empty_cache()
        


    
    for new_ppl, new_uncond_ppl, orig_idx in zip(model_ppls, unconditional_ppls, idxs):
        baseline_ppl = generated_solution_ppl[orig_idx]
        unconditional_baseline_ppl = unconditional_solution_ppl[orig_idx]
        baseline_improvement = (baseline_ppl - new_ppl) / baseline_ppl
        unconditional_improvement = (unconditional_baseline_ppl - new_uncond_ppl) / unconditional_baseline_ppl

        reward[orig_idx] = max(0.0, baseline_improvement) - (unconditional_lambda * max(0.0,unconditional_improvement))
        ppl_improvement_for_recording[orig_idx] = baseline_improvement
        unconditional_improvement_for_recording[orig_idx] = unconditional_improvement
        if clip_reward:
            reward[orig_idx] = max(0, reward[orig_idx])
    
    for j, i in enumerate(idxs):
        # print("problem")
        # print(problem[i])
        # print("abstraction")
        # print(abstractions[i])
        # print("baseline ppl")
        # print(generated_solution_ppl[i])
        # print("model ppl")
        # print(model_ppls[j])
        # print("reward")
        # print(reward[i])
        model_ppls_for_recording[i] = model_ppls[j]
        # ppl_improvement_for_recording[]

        # print("prefix")
        # print(prefixes[i])
        # print("full text")
        # print(full_prompts[i])
    
    if rollout_logger is not None:
        rows = []
        for i in range(n):
            rows.append({
                "problem": problem[i],
                "prompt": prompts[i],
                "completion": completions[i],
                "abstraction": abstractions[i],
                "solver_user_prompt": solving_prompts[i],
                "prefix": prefixes[i],
                "full_ppl_prompt": full_prompts_for_recording[i],
                "baseline_ppl": generated_solution_ppl[i],
                "model_ppl": model_ppls_for_recording[i],
                "ppl_improvement": ppl_improvement_for_recording[i],
                "unconditional_improvement": unconditional_improvement_for_recording[i],
                "reward": reward[i]
            })
        rollout_logger.add_rows(rows)

    return reward


# --- add helper to find latest checkpoint (anywhere above __main__) ---
def find_latest_checkpoint(run_dir: str) -> str | None:
    p = Path(run_dir)
    if not p.exists():
        return None
    ckpts = []
    for d in p.iterdir():
        if d.is_dir() and d.name.startswith("checkpoint-"):
            try:
                step = int(d.name.split("-")[-1])
                ckpts.append((step, str(d)))
            except Exception:
                pass
    if not ckpts:
        return None
    ckpts.sort(key=lambda x: x[0])
    return ckpts[-1][1]

def maybe_load_wandb_id(run_dir: str) -> str | None:
    # if you store it yourself, read it; otherwise try wandb's run file if present
    # simplest: put wandb_id.txt in run_dir once at start of training
    wid = Path(run_dir) / "wandb_id.txt"
    if wid.exists():
        return wid.read_text().strip()
    return None

def save_grpo_config(config, output_dir: str):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # best-effort -> plain JSON-serializable dict
    if hasattr(config, "to_dict"):
        d = config.to_dict()
    elif is_dataclass(config):
        d = asdict(config)
    else:
        d = dict(config.__dict__)

    # handle any non-JSON types (e.g., Path) by stringifying
    def default(o): return str(o)

    (out / "grpo_config.json").write_text(
        json.dumps(d, indent=2, sort_keys=True, default=default)
    )


def get_arguments():
    parser = ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--save_rollouts", action="store_true")
    parser.add_argument("--baseline_no_sft", action="store_true")
    parser.add_argument("--baseline_thinking", action="store_true")
    parser.add_argument("--baseline_few_shot", action="store_true")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--with_reasoning", action="store_true")
    parser.add_argument("--no_think", action="store_true")
    parser.add_argument("--clip_reward", action="store_true")
    parser.add_argument("--completion_length", type=int, default=256)
    parser.add_argument("--subtract_compression_ppl", action="store_true")
    parser.add_argument("--unconditional_lambda", type=float, default=0.0)
    
    parser.add_argument("--run_dir", type=str, default=None,
                        help="If set, use this as output_dir (for resume or manual naming).")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from the latest checkpoint in --run_dir (or auto-derived).")
    parser.add_argument("--resume_checkpoint", type=str, default=None,
                        help="Explicit checkpoint path (overrides auto-latest).")
    parser.add_argument("--wandb_resume", action="store_true",
                        help="Resume the same W&B run if wandb_id is found in run_dir.")

    return parser


if __name__ == "__main__":
    args = get_arguments().parse_args()
    print("ARGS")
    print(args)
    local_rank, rank, world_size = get_distributed_context()
    is_main_process = rank == 0
    process_device = get_process_device(local_rank)
    dtype = torch.bfloat16
    if is_main_process:
        print(f"Distributed context: rank={rank} local_rank={local_rank} world_size={world_size} device={process_device}")

    MODEL_NAME = "Qwen/Qwen3-1.7B"
    if args.baseline_no_sft or args.baseline_few_shot:
        STRATEGY_MODEL_NAME = MODEL_NAME
    else:
        if args.with_reasoning:
            STRATEGY_MODEL_NAME = "sft_qwen_1_7B_from_trace_chkpt4728"
        else:
            STRATEGY_MODEL_NAME = "sft_qwen_1_7B_1_2M_checkpoint118155" #"sft_qwen_1_7B_chkpt7764" #"sft_qwen_1_7B_1_2M_checkpoint118155"
    
    ppl_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
    )
    ppl_model.to(process_device)
    ppl_model.eval()
    # MODEL_NAME = "sft_qwen_1_7B_chkpt7764"
    # LORA_PATH = "/scratch/rst306/action_abstractions/action_abstraction/sft_models/Qwen3_1_7B-abstraction_generation/20260115_092723/checkpoint-118155"
    # --- CHANGE: force the trainable model onto the (only visible) GPU0
    # base_model = AutoModelForCausalLM.from_pretrained(
    #     MODEL_NAME,
    #     torch_dtype=torch.bfloat16,
    #     device_map={"": 0},
    # )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


    strategy_model = AutoModelForCausalLM.from_pretrained(
        STRATEGY_MODEL_NAME,
        torch_dtype=dtype,
    )

    strategy_model.train()

    # -------------------------------------------------------------------
    if args.baseline_few_shot:
        if args.baseline_thinking:
            thinking=True
        else:
            thinking=False

        with open("/scratch/rst306/action_abstractions/action_abstraction/prompt_templates/abstraction_generation_succinct_few_shot_prompt_with_think.txt", "r") as fp:
            sft_prompt_template = fp.read()

        def add_prompt_column(ex):
            solution = ex["correct_solution"]
            prompt = sft_prompt_template.replace("{{PROBLEM}}", ex["problem"]).replace("{{REASONING}}", remove_think_blocks(solution))
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=thinking,  # Set to False to strictly disable thinking
            )
            ex["prompt"] = text
            return ex

    if args.baseline_no_sft:
        if args.baseline_thinking:
            thinking=True
        else:
            thinking=False
        if args.with_reasoning:
            with open("/scratch/rst306/action_abstractions/action_abstraction/prompt_templates/sft_problem_reasoning_abstraction_generation_prompt.txt", "r") as fp:
                sft_prompt_template = fp.read()
            # with open("/scratch/rst306/action_abstractions/action_abstraction/prompt_templates/abstraction_generation_succinct_few_shot_prompt_with_think.txt", 'r') as fp:
            #     sft_prompt_template = fp.read()

            def add_prompt_column(ex):
                solution = ex["correct_solution"]
                prompt = sft_prompt_template.replace("{{PROBLEM}}", ex["problem"]).replace("{{REASONING}}", remove_think_blocks(solution))
                messages = [{"role": "user", "content": prompt}]
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=thinking,  # Set to False to strictly disable thinking
                )
                ex["prompt"] = text
                return ex

        else:
            raise Exception("Not implemented")
            # with open("/scratch/rst306/action_abstractions/action_abstraction/prompt_templates/sft_abstraction_generation.txt", "r") as fp:
            #     sft_prompt_template = fp.read()
            # with open("/scratch/rst306/action_abstractions/action_abstraction/prompt_templates/abstraction_generation_succint_few_shot_prompt_without_think.txt", 'r') as fp:
            #     sft_prompt_template = fp.read()
                
            # def add_prompt_column(ex):
            #     prompt = sft_prompt_template.replace("{{PROBLEM}}", ex["problem"])
            #     messages = [{"role": "user", "content": prompt}]
            #     text = tokenizer.apply_chat_template(
            #         messages,
            #         tokenize=False,
            #         add_generation_prompt=True,
            #         enable_thinking=thinking,  # Set to False to strictly disable thinking
            #     )
            #     ex["prompt"] = text
            #     return ex
    

    dataset = load_from_disk(args.dataset)
    print(dataset)

    cols_to_keep = ["problem", "correct_solution", "generated_solution_ppl", "unconditional_solution_ppl"]
    cols_to_drop = [c for c in dataset["train"].column_names if c not in cols_to_keep]
    dataset["train"] = dataset["train"].remove_columns(cols_to_drop)

    dataset["train"] = dataset["train"].map(add_prompt_column)

    dataset["test"] = dataset["test"].remove_columns(cols_to_drop)

    dataset["test"] = dataset["test"].map(add_prompt_column)

    # dataset['train'] = dataset["train"].select(list(range(1)))

    print("dataset")
    print(dataset["train"][0])

    if is_main_process:
        wandb.init(project="abstraction_grpo_ppl")

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
    now = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.baseline_no_sft:
        experiment_type = "qwen3_1_7b_deepscaler_easy_no_sft"
    elif args.baseline_few_shot:
        experiment_type = "qwen3_1_7b_deepscaler_easy_no_sft_few_shot"
    else:
        experiment_type = "qwen3_1_7b_deepscaler_easy"

    completion_length = args.completion_length


    # --- after now/experiment_type are computed, choose output_dir ---
    base_out = f"/scratch/rst306/action_abstractions/action_abstraction/grpo_runs/ppl/{experiment_type}/{now}"
    output_dir = args.run_dir if args.run_dir is not None else base_out

    # If resume requested without run_dir, you can’t infer a prior run safely.
    # So require run_dir for resume (or pass resume_checkpoint explicitly).
    if args.resume and args.run_dir is None and args.resume_checkpoint is None:
        raise ValueError("--resume requires --run_dir (or set --resume_checkpoint).")
    print(f"Running with output dir {output_dir}")
    config = GRPOConfig(
        output_dir=output_dir,
        num_train_epochs=2,
        # num_train_epochs=100000000,
        adam_beta2=0.95,
        learning_rate=args.lr,
        report_to="wandb",
        bf16=True,
        temperature=args.temperature,
        top_p=0.95,
        top_k=20,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        # per_device_train_batch_size=2,
        # per_device_eval_batch_size=8,
        gradient_accumulation_steps=6,
        num_generations=4,
        num_generations_eval=1,
        # num_generations=4,
        # generation_batch_size=32,
        # generation_batch_size=64,
        gradient_checkpointing=True,
        # eval_strategy="steps",
        do_eval=False,
        # eval_steps=200,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=1,
        # save_steps=200,          # <-- pick your cadence
        save_total_limit=4,     # optional
        seed=args.seed,
        data_seed=args.seed,
        max_completion_length=completion_length
    )
    print("CONFIG")
    print(config)

    # --- W&B init: resume same run id if desired ---
    if is_main_process:
        wandb_kwargs = {"project": "abstraction_grpo_ppl"}
        if args.wandb_resume:
            wid = maybe_load_wandb_id(config.output_dir)
            if wid:
                wandb_kwargs.update({"id": wid, "resume": "must"})
        run = wandb.init(**wandb_kwargs)

        # store id for later resume
        wid_path = Path(config.output_dir) / "wandb_id.txt"
        wid_path.parent.mkdir(parents=True, exist_ok=True)
        wid_path.write_text(run.id)

    if args.save_rollouts:
        rollout_root = os.path.join(config.output_dir, "rollouts")
        rollout_logger = RolloutLogger(rollout_root, load_existing=args.resume or (args.resume_checkpoint is not None))
    else:
        rollout_logger = None

    def dummy_reward_fn(prompts, completions, completion_ids, problem, correct_solution, generated_solution_ppl, unconditional_solution_ppl, model, tokenizer, batch_size, no_think=False, clip_reward=False, rollout_logger=None, subtract_compression_ppl=False, unconditional_lambda=0, **kwargs):
        rewards = [0] * len(completions)
        # for i, c in enumerate(completions):
        #     if i % 4 == 0:
        #         rewards.append(1)
        #     else:
        return rewards


    reward_fn = partial(ppl_reward_function, model=ppl_model, tokenizer=tokenizer, batch_size=2, rollout_logger=rollout_logger, no_think=args.no_think, clip_reward=args.clip_reward, unconditional_lambda=args.unconditional_lambda)

    reward_fn.__name__ = "solver_reward"

    trainer = GRPOTrainer(
        model=strategy_model,
        reward_funcs=[reward_fn],
        args=config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        processing_class=tokenizer,
        peft_config=lora_config
    )


    if args.save_rollouts:
        trainer.add_callback(RolloutSaveCallback(rollout_logger))
    
    resume_path = None
    if args.resume_checkpoint is not None:
        resume_path = args.resume_checkpoint
    elif args.resume:
        resume_path = find_latest_checkpoint(config.output_dir)

    trainer.train(resume_from_checkpoint=resume_path)