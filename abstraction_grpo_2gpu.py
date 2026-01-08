# --- NEW (must be at the very top, before torch/transformers imports)
import os
TRAIN_GPU = "0"
SOLVER_GPU = "1"

# def _vllm_solver_worker(req_q, resp_q, model_name, gpu_mem_util, sp_kwargs):
#     import os
#     os.environ["CUDA_VISIBLE_DEVICES"] = SOLVER_GPU
#     os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"  # critical

#     from vllm import LLM, SamplingParams
#     llm = LLM(model=model_name, gpu_memory_utilization=gpu_mem_util, tensor_parallel_size=1)
#     sp = SamplingParams(**sp_kwargs)

#     while True:
#         texts = req_q.get()
#         if texts is None:
#             break
#         outs = llm.generate(texts, sp)
#         resp_q.put([o.outputs[0].text for o in outs])

# class SolverClient:
#     def __init__(self, req_q, resp_q):
#         self.req_q = req_q
#         self.resp_q = resp_q
#     def generate_texts(self, texts):
#         self.req_q.put(texts)
#         return self.resp_q.get()
binary_judge_lora_path = "/scratch/rst306/action_abstractions/action_abstraction/sft_models/binary_judge/Qwen3_1_7B-abstraction_generation/checkpoint-768"
with open("/scratch/rst306/action_abstractions/action_abstraction/prompt_templates/binary_judge_sft_template.txt", 'r') as fp:
    lines = fp.readlines()

binary_judge_prompt_template = "".join(lines)

def make_binary_judge_prompt(problem: str, abstraction: str, solution_trace: str, template: str) -> str:
    out = template.replace("{{PROBLEM}}",problem).replace("{{ABSTRACTION}}", abstraction).replace("{{SOLUTION}}", solution_trace)
    return out

def parse_binary_judge_output(text: str) -> float:
    # Return 1.0 if the judge says it used the abstraction, else 0.0.
    import re
    m = re.search(r"<judgement>(.*?)</judgement>", text, flags=re.DOTALL)
    return float(m.group(1)) if m else None

def _vllm_solver_worker(req_q, resp_q, model_name, gpu_mem_util, sp_solve_kwargs, sp_judge_kwargs, judge_lora_path):
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = SOLVER_GPU
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"  # critical

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    # One engine; base requests omit lora_request, judge requests include it.
    llm = LLM(
        model=model_name,
        gpu_memory_utilization=gpu_mem_util,
        tensor_parallel_size=1,
        enable_lora=True,
    )

    sp_solve = SamplingParams(**sp_solve_kwargs)
    sp_judge = SamplingParams(**sp_judge_kwargs)

    judge_lora_req = LoRARequest("binary_judge", 1, judge_lora_path)

    while True:
        req = req_q.get()
        if req is None:
            break

        # Back-compat: if old code sends a list, treat as solve.
        if isinstance(req, list):
            kind, texts = "solve", req
        else:
            kind = req.get("kind", "solve")
            texts = req["texts"]

        if kind == "solve":
            outs = llm.generate(texts, sp_solve)
            resp_q.put([o.outputs[0].text for o in outs])
        elif kind == "judge":
            outs = llm.generate(texts, sp_judge, lora_request=judge_lora_req)
            resp_q.put([o.outputs[0].text for o in outs])
        else:
            raise ValueError(f"Unknown request kind: {kind}")

class SolverClient:
    def __init__(self, req_q, resp_q):
        self.req_q = req_q
        self.resp_q = resp_q

    def generate_texts(self, texts):
        # base solve
        self.req_q.put({"kind": "solve", "texts": texts})
        return self.resp_q.get()

    def judge_texts(self, texts):
        # LoRA judge
        self.req_q.put({"kind": "judge", "texts": texts})
        return self.resp_q.get()


# -------------------------------------------------------------------
# (keep your existing imports below; torch/transformers now only see GPU0)
from datasets import load_from_disk
from trl import GRPOTrainer, GRPOConfig
from problem_solving import make_problem_solving_prompt
from abstraction_generation_processing import extract_abstraction_from_tag
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
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

class GPUBusySpinner:
    """
    Runs a GEMM loop on GPU0 in a background thread while `active` is set.
    This keeps GPU utilization/clocks high during the CPU-blocking reward wait.
    """
    def __init__(self, device: int = 0, dim: int = 2048, iters_per_sync: int = 8, dtype=torch.bfloat16, stream_priority: int = 2):
        self.device = device
        self.dim = dim
        self.iters_per_sync = iters_per_sync
        self.dtype = dtype
        self.stream_priority = stream_priority

        self._active = threading.Event()
        self._stop = threading.Event()
        self._ready = threading.Event()

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._ready.wait()

    def activate(self):
        self._active.set()

    def deactivate(self):
        self._active.clear()

    def close(self):
        self._stop.set()
        self._active.set()  # wake thread so it can exit
        self._thread.join()

    def _run(self):
        torch.cuda.set_device(self.device)

        # Create a low-priority stream (best-effort; some setups ignore priority)
        try:
            stream = torch.cuda.Stream(device=self.device, priority=self.stream_priority)
        except TypeError:
            stream = torch.cuda.Stream(device=self.device)

        # Pre-allocate tensors once
        with torch.cuda.device(self.device):
            a = torch.randn((self.dim, self.dim), device="cuda", dtype=self.dtype)
            b = torch.randn((self.dim, self.dim), device="cuda", dtype=self.dtype)

        self._ready.set()

        while not self._stop.is_set():
            # Sleep when not active
            if not self._active.is_set():
                time.sleep(0.001)
                continue

            # Keep the GPU busy with BF16 GEMMs
            with torch.cuda.stream(stream):
                x = a
                for _ in range(self.iters_per_sync):
                    x = x @ b
            stream.synchronize()

class RolloutLogger:
    """
    Accumulates per-completion rollout records and writes them as a HF Dataset.
    Saves:
      - {root}/latest              (overwritten each save)
      - {root}/step-{global_step}  (snapshot per checkpoint)
    """
    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        os.makedirs(self.root_dir, exist_ok=True)
        self._lock = threading.Lock()
        self._rows = []
        self._step = 0

    def set_step(self, step: int):
        self._step = int(step)

    def add_rows(self, rows):
        with self._lock:
            # stamp step at insertion time
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
        snap_dir = os.path.join(self.root_dir, f"step-{int(step)}")
        with self._lock:
            self._save_dir(latest_dir)
            self._save_dir(snap_dir)

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



# MODEL_NAME = "Qwen/Qwen3-8B"
# LORA_PATH = "/scratch/rst306/action_abstractions/action_abstraction/qwen-abstraction_generation/checkpoint-1070"
MODEL_NAME = "Qwen/Qwen3-1.7B"
LORA_PATH = "/scratch/rst306/action_abstractions/action_abstraction/sft_models/Qwen3_1_7B-abstraction_generation/checkpoint-7764"
# --- CHANGE: force the trainable model onto the (only visible) GPU0
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map={"": 0},
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

strategy_model = PeftModel.from_pretrained(base_model, LORA_PATH)
strategy_model = strategy_model.to(base_model.device)
strategy_model.train()

# def extract_answer_from_solution(text):
#     m = list(re.finditer(r'\\boxed\{', text))
#     if not m:
#         return None
#     start = m[-1].end()
#     depth = 1
#     i = start
#     while i < len(text) and depth > 0:
#         c = text[i]
#         if c == '{':
#             depth += 1
#         elif c == '}':
#             depth -= 1
#         i += 1
#     if depth != 0:
#         return None
#     return text[start:i-1]

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

with open("/scratch/rst306/action_abstractions/action_abstraction/prompt_templates/baseline_problem_solving.txt", 'r') as fp:
    lines = fp.readlines()

problem_solving_prompt_template = "".join(lines)

def build_baseline_solving_prompt(problem):

    prompt = problem_solving_prompt_template.replace("{{PROBLEM}}", problem)
    return prompt

@torch.no_grad()
def problem_solving_reward_fn(
    prompts, completions, completion_ids, problem, gt_answer,
    model, tokenizer, rollout_logger=None, baseline_reference_reward=False, **kwargs
):
    n = len(completions)
    rewards = [0.0] * n
    print("GT ANS")
    print(gt_answer)
    # per-item fields we’ll fill
    abstractions = [None] * n
    solving_prompts = [None] * n
    solver_inputs = [None] * n
    solver_outputs = [None] * n
    pred_answers = [None] * n

    # judge logging
    judge_user_prompts = [None] * n
    judge_inputs = [None] * n
    judge_outputs = [None] * n
    adheres = [None] * n  # float 0/1

    # baseline reward
    baseline_prompts = [None] * n
    baseline_inputs = [None] * n
    baseline_outputs = [None] * n
    baseline_pred_answers = [None] * n
    baseline_corrects = [None] * n  # bool/float


    # rewards = []
    idxs = []
    texts = []
    targets = []

    if baseline_reference_reward:
        problem_to_gt_answer = {}
        for prob, gt in zip(problem, gt_answer):
            if prob not in problem_to_gt_answer:
                problem_to_gt_answer[prob] = gt

        baseline_texts = []
        problem_to_baseline_solution = {}
        problem_to_baseline_prompt = {}
        problem_to_baseline_output = {}
        problem_to_baseline_input = {}
        problem_to_baseline_pred_answer = {}
        problem_to_baseline_correct = {}
        
        unique_problems = list(set(problem))

        # to do, check len(problem), see if it is unique for each completion (probably not) and then rewrite this
        for prob in unique_problems:
            baseline_prompt = build_baseline_solving_prompt(prob)
            messages = [{"role": "user", "content": baseline_prompt}]
            baseline_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )

            baseline_texts.append(baseline_text)
            print("baseline")
            print(baseline_text)

            problem_to_baseline_prompt[prob] = baseline_prompt
            problem_to_baseline_input[prob] = baseline_text


        baseline_solutions = model.generate_texts(baseline_texts)
        for i, sol_text in enumerate(baseline_solutions):
            prob = unique_problems[i]
            
            problem_to_baseline_output[prob] = sol_text
            
            pred = extract_answer_from_solution(sol_text)
            problem_to_baseline_pred_answer[prob] = pred

            gt_ans = problem_to_gt_answer[prob]
            target = "\\boxed{" + gt_ans + "}"
            if pred is not None and verify(parse(target), parse(pred)):
                problem_to_baseline_correct[prob] = 1.0
            else:
                problem_to_baseline_correct[prob] = 0.0

            print("baseline problem")
            print(prob)
            print("baseline output")
            print(sol_text)
            print("baseline pred answer")
            print(pred)
            print("correct/incorrect")
            print(problem_to_baseline_correct[prob])
            print("-"*10)


    for i, strategy in enumerate(completions):
        prob = problem[i]
        # target = str(gt_answer[i]).strip()
        target = "\\boxed{" + gt_answer[i] + "}"
        
        abstraction = extract_abstraction_from_tag(strategy)
        abstractions[i] = abstraction
        if abstraction is None or abstraction == "":
            rewards[i] = 0.0
            adheres[i] = 0.0
            continue

        solving_prompt = make_problem_solving_prompt(prob, abstraction)
        solving_prompts[i] = solving_prompt
        messages = [{"role": "user", "content": solving_prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )

        idxs.append(i)
        texts.append(text)
        targets.append(target)
        solver_inputs[i] = text

    if len(texts) > 0:
        # --- CHANGE: call the solver client (GPU1 process)
        print("TARGETS")
        print(targets)

        print("IDXS")
        print(idxs)
        if GPU_SPINNER is not None:
            GPU_SPINNER.activate()
        try:
            problem_solutions = model.generate_texts(texts)  # blocks on GPU1 solver
        finally:
            if GPU_SPINNER is not None:
                GPU_SPINNER.deactivate()

        for j, i in enumerate(idxs):
            sol_text = problem_solutions[j]
            solver_outputs[i] = sol_text

            pred = extract_answer_from_solution(sol_text)
            pred_answers[i] = pred

            # if pred is not None and pred == targets[j]:
            if pred is not None and verify(parse(targets[j]), parse(pred)):
                rewards[i] = 1.0
            else:
                rewards[i] = 0.0

        # NEW: build judge batch over the same idxs (only where abstraction exists)
        judge_idxs, judge_texts = [], []
        for j, i in enumerate(idxs):
            if not abstractions[i]:
                adheres[i] = 0.0
                continue

            judge_prompt = make_binary_judge_prompt(
                problem=problem[i],
                abstraction=abstractions[i],
                solution_trace=solver_outputs[i],
                template=binary_judge_prompt_template,
            )
            judge_user_prompts[i] = judge_prompt

            judge_msg = [{"role": "user", "content": judge_prompt}]
            judge_text = tokenizer.apply_chat_template(
                judge_msg,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            judge_inputs[i] = judge_text
            judge_idxs.append(i)
            judge_texts.append(judge_text)

        if len(judge_texts) > 0:
            if GPU_SPINNER is not None:
                GPU_SPINNER.activate()
            try:
                judge_resps = model.judge_texts(judge_texts)  # GPU1 LoRA judge
            finally:
                if GPU_SPINNER is not None:
                    GPU_SPINNER.deactivate()

            for k, i in enumerate(judge_idxs):
                jtxt = judge_resps[k]
                judge_outputs[i] = jtxt
                adheres[i] = parse_binary_judge_output(jtxt)
                print("abstraction")
                print(abstractions[i])
                print("solution trace")
                print(solver_outputs[i])
                print("solution")
                print(pred_answers[i])
                print("target")
                print(targets[k])
                print("judgement string")
                print(jtxt)
                print("parsed judgement")
                print(adheres[i])
                print("original reward")
                print(rewards[i])
                print("-" * 10)


        # combine: require both correct + adherent
        for i in idxs:
            rewards[i] = float(rewards[i]) * float(adheres[i] if adheres[i] is not None else 0.0)
            if baseline_reference_reward:
                print("baseline ref reward")
                print("orig reward", rewards[i])
                prob = problem[i]
                baseline_correct = problem_to_baseline_correct[prob]
                print("baseline correct", baseline_correct)
                rewards[i] = rewards[i] * float(1.0 - baseline_correct)
                print("final reward", rewards[i])

                baseline_corrects[i] = baseline_correct
                baseline_pred_answers[i] = problem_to_baseline_pred_answer[prob]
                baseline_prompts[i] = problem_to_baseline_prompt[prob]
                baseline_outputs[i] = problem_to_baseline_output[prob]
                baseline_inputs[i] = problem_to_baseline_input[prob]


    if rollout_logger is not None:
        rows = []
        for i in range(n):
            rows.append({
                "problem": problem[i],
                "prompt": prompts[i],
                "completion": completions[i],
                "gt_answer": str(gt_answer[i]).strip(),
                "abstraction": abstractions[i],
                "solver_user_prompt": solving_prompts[i],
                "solver_input_text": solver_inputs[i],
                "solver_output_text": solver_outputs[i],
                "pred_answer": pred_answers[i],
                # NEW judge fields
                "judge_user_prompt": judge_user_prompts[i],
                "judge_input_text": judge_inputs[i],
                "judge_output_text": judge_outputs[i],
                "abstraction_adheres": float(adheres[i] if adheres[i] is not None else 0.0),
                "reward": float(rewards[i]),
                # baseline reference fields
                "baseline_prompt": baseline_prompts[i],
                "baseline_input_text": baseline_inputs[i],
                "baseline_output_text": baseline_outputs[i],
                "baseline_pred_answer": baseline_pred_answers[i],
                "baseline_correct": baseline_corrects[i],
            })
        rollout_logger.add_rows(rows)
    print("STEP")
    return rewards


def get_arguments():
    parser = ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--baseline_reference_reward", action="store_true")
    return parser
# -------------------------- NEW: start solver process + client --------------------------

GPU_SPINNER = None


if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = TRAIN_GPU  # training proc uses GPU0 only

    import multiprocessing as mp
    ctx = mp.get_context("spawn")  # spawn only for the solver proc

    req_q = ctx.Queue()
    resp_q = ctx.Queue()

    sp_solve_kwargs = dict(temperature=0.6, top_p=0.95, top_k=20, max_tokens=8000)
    # judge should be short + deterministic
    sp_judge_kwargs = dict(temperature=0.0, top_p=1.0, max_tokens=128)
    solver_proc = ctx.Process(
        target=_vllm_solver_worker,
        args=(req_q, resp_q, MODEL_NAME, 0.5, sp_solve_kwargs, sp_judge_kwargs, binary_judge_lora_path),
    )

    # solver_proc = ctx.Process(
    #     target=_vllm_solver_worker,
    #     args=(req_q, resp_q, MODEL_NAME, 0.5, sp_kwargs),
    # )
    solver_proc.start()

    solver_client = SolverClient(req_q, resp_q)

    # -------------------------------------------------------------------
    args = get_arguments().parse_args()
    print("ARGS")
    print(args)
    with open("/scratch/rst306/action_abstractions/action_abstraction/prompt_templates/sft_abstraction_generation.txt", "r") as fp:
        sft_prompt_template = fp.read()

    def add_prompt_column(ex):
        ex["prompt"] = sft_prompt_template.replace("{{PROBLEM}}", ex["problem"])
        return ex

    # dataset = load_from_disk("openmathreasoning_500_subset").select(list(range(10)))
    # dataset = load_from_disk("deepscaler_easy_train_test")
    dataset = load_from_disk("deepscaler_easy_train_small_test")
    print(dataset)
    # cols_to_keep = ["expected_answer", "problem"]
    cols_to_keep = ["answer", "problem"]
    cols_to_drop = [c for c in dataset["train"].column_names if c not in cols_to_keep]
    dataset["train"] = dataset["train"].remove_columns(cols_to_drop)
    dataset["train"] = dataset["train"].rename_columns({"answer": "gt_answer"})
    dataset["train"] = dataset["train"].map(add_prompt_column)

    dataset["test"] = dataset["test"].remove_columns(cols_to_drop)
    dataset["test"] = dataset["test"].rename_columns({"answer": "gt_answer"})
    dataset["test"] = dataset["test"].map(add_prompt_column)

    print("Data")
    print(dataset["train"][0])
    print(dataset["test"][0])
    indices = random.sample(range(len(dataset["train"])), k=8)
    dataset["train"] = dataset["train"].select(indices)

    wandb.init(project="abstraction_grpo")

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

    config = GRPOConfig(
        output_dir=f"/scratch/rst306/action_abstractions/action_abstraction/grpo_runs/qwen3_1_7b_deepscaler_easy/{now}",
        num_train_epochs=10000000,
        adam_beta2=0.95,
        learning_rate=args.lr,
        report_to="wandb",
        bf16=True,
        # temperature=0.8,
        # per_device_train_batch_size=16,
        # per_device_eval_batch_size=16,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=1,
        num_generations=4,
        generation_batch_size=32,
        gradient_checkpointing=True,
        # eval_strategy="steps",
        # do_eval=True,
        # eval_steps=200,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=1,
        # save_steps=200,          # <-- pick your cadence
        # save_total_limit=4,     # optional
        seed=args.seed,
        data_seed=args.seed
    )
    print("CONFIG")
    print(config)

    rollout_logger = RolloutLogger(os.path.join(config.output_dir, "rollouts"))

    reward_fn = partial(
        problem_solving_reward_fn,
        model=solver_client,
        tokenizer=tokenizer,
        rollout_logger=rollout_logger,
        baseline_reference_reward=args.baseline_reference_reward,
    )

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

    GPU_SPINNER = GPUBusySpinner(device=0, dim=2048, iters_per_sync=8, dtype=torch.bfloat16)

    trainer.add_callback(RolloutSaveCallback(rollout_logger))
    trainer.train()

    # --- NEW: clean shutdown
    req_q.put(None)
    solver_proc.join()
    GPU_SPINNER.close()

