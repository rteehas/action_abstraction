from datasets import load_dataset, load_from_disk
from trl import GRPOTrainer
from problem_solving import make_problem_solving_prompt
from abstraction_generation_processing import extract_abstraction_from_tag
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from trl import GRPOTrainer, GRPOConfig
from functools import partial
import re
import wandb
from vllm import LLM, SamplingParams


MODEL_NAME = "Qwen/Qwen3-8B"          # example
LORA_PATH = "/scratch/rst306/action_abstractions/action_abstraction/qwen-abstraction_generation/checkpoint-1070"
sampling_params = SamplingParams(temperature=0.6, top_p=0.95, top_k=20, max_tokens=32768)

# Initialize the vLLM engine
llm = LLM(model=MODEL_NAME, gpu_memory_utilization=0.5)

# load base model once
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    # device_map="auto",
).to("cuda:0")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# wrap with LoRA for the "strategy" model
strategy_model = PeftModel.from_pretrained(base_model, LORA_PATH)
strategy_model = strategy_model.to(base_model.device)
strategy_model.train()  # GRPO will update the LoRA params only

# def extract_answer_from_solution(text):
#     m = re.findall(r'\\boxed\{([^}]*)\}', text)
#     if len(m) > 0:
#         return m[-1]
#     else:
#         return None
def extract_answer_from_solution(text):
    # find the last \boxed{ ... }
    m = list(re.finditer(r'\\boxed\{', text))
    if not m:
        return None

    start = m[-1].end()  # position just after '\boxed{'
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
        # unbalanced braces
        return None

    # content inside the outermost \boxed{ ... }
    return text[start:i-1]

@torch.no_grad()
def problem_solving_reward_fn(
    prompts,        # list[str] – the prompts sent to the strategy model
    completions,    # list[str] – strategy outputs (the things we trained with GRPO)
    completion_ids, # usually not needed unless you use it for logging
    problem,        # list[str] – original math problems (extra column in dataset)
    gt_answer,      # list[str] – ground-truth answers (same length as problem)
    model: LLM,
    tokenizer: AutoTokenizer,
    **kwargs
):
    """
    Returns a list of rewards (float) of same length as `completions`.
    """
    print("prompts")
    for prompt in prompts:
        print(prompt)

    print("completions")
    for i, comp in enumerate(completions):
        print(i, comp)
    
    print("problems")
    for i, prob in enumerate(problem):
        print(i, prob)
    rewards = []
    
    texts = []
    for i, strategy in enumerate(completions):
        prob = problem[i]
        target = str(gt_answer[i]).strip()

        # 1. Extract abstraction / strategy object from the strategy text
        abstraction = extract_abstraction_from_tag(strategy)
        print(f"Abstraction = {abstraction}")
        if abstraction is None:
            rewards.append(0.0)
            continue

        # 2. Build solving prompt (problem + abstraction)
        solving_prompt = make_problem_solving_prompt(prob, abstraction)
        messages = [{"role": "user", "content": solving_prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,  # Set to False to strictly disable thinking
        )
        texts.append(text)
        # print("text", text)
        # inputs = tokenizer(
        #     text,
        #     return_tensors="pt",
        #     padding=False,
        #     truncation=True,
        # ).to(device)

        # # 3. Use the same model but **without** LoRA → base solver
        # if hasattr(model, "disable_adapter"):
        #     model.disable_adapter()        # temporarily turn off LoRA
        # outputs = model.generate(
        #     **inputs,
        #     max_new_tokens=4000,
        #     do_sample=True,
        # )
        # if hasattr(model, "enable_adapter"):
        #     model.enable_adapter()         # turn LoRA back on for strategy training

        # # 4. Decode only the generated continuation
        # gen_ids = outputs[0][inputs["input_ids"].shape[-1]:]
        # solution_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        # print(solution_text)
        # # 5. Extract the predicted answer and compute reward
        # pred_answer = extract_answer_from_solution(solution_text)
        # print(f"Pred answer = {pred_answer}, Target = {target}")
        # if pred_answer is None:
        #     rewards.append(0.0)
        # elif pred_answer == target:
        #     rewards.append(1.0)  # exact-match reward
        # else:
        #     rewards.append(0.0)
    sampling_params = SamplingParams(temperature=0.6, top_p=0.95, top_k=20, max_tokens=32768)
    outputs = model.generate(texts, sampling_params)
    problem_solutions = []
    for output in outputs:
        generated_text = output.outputs[0].text
        problem_solutions.append(generated_text)
    
    for solution_text in problem_solutions:
        print(solution_text)
        pred_answer = extract_answer_from_solution(solution_text)
        print(f"Pred answer = {pred_answer}, Target = {target}")
        print("*" * 20)
        if pred_answer is None:
            rewards.append(0.0)
        elif pred_answer == target:
            rewards.append(1.0)  # exact-match reward
        else:
            rewards.append(0.0)

    return rewards

def dummy_reward_fn(prompts, completions, **kwargs):
    return [1.0] * len(prompts)

with open("/scratch/rst306/action_abstractions/action_abstraction/prompt_templates/sft_abstraction_generation.txt", 'r') as fp:
    lines = fp.readlines()

sft_prompt_template = "".join(lines)

def add_prompt_column(ex):
    problem = ex["problem"]
    prompt = sft_prompt_template.replace("{{PROBLEM}}",problem)
    ex["prompt"] = prompt
    return ex

dataset = load_from_disk("openmathreasoning_500_subset").select(list(range(10)))
cols_to_keep = ["expected_answer", "problem"]
cols_to_drop = [c for c in dataset.column_names if c not in cols_to_keep]
dataset = dataset.remove_columns(cols_to_drop)
dataset = dataset.rename_columns({"expected_answer": "gt_answer"})
dataset = dataset.map(add_prompt_column)


wandb.init(project="abstraction_grpo")

config = GRPOConfig(
    learning_rate=1e-5,
    report_to="wandb",      
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    num_generations=2,  # group size in GRPO
    generation_batch_size=2,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    logging_steps=1
)

reward_fn = partial(
    problem_solving_reward_fn,
    model=llm,
    tokenizer=tokenizer,
)
reward_fn.__name__ = "solver_reward"

trainer = GRPOTrainer(
    model=strategy_model,      # this is the LoRA-augmented strategy model
    reward_funcs=[reward_fn],
    args=config,
    train_dataset=dataset,
    processing_class=tokenizer,
)

trainer.train()
