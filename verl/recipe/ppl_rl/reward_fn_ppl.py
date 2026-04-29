from __future__ import annotations

import asyncio
import json
import math
import re
from pathlib import Path

import aiohttp


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_HINT_TEMPLATE = str(REPO_ROOT / "prompt_templates" / "hint_conditioned_problem_solving_rich_v1.txt")
_TEMPLATE_CACHE: dict[str, str] = {}


def load_template(path: str) -> str:
    cached = _TEMPLATE_CACHE.get(path)
    if cached is not None:
        return cached
    template = Path(path).read_text()
    _TEMPLATE_CACHE[path] = template
    return template


def render_template(template: str, problem: str, abstraction: str) -> str:
    return template.replace("{{PROBLEM}}", problem).replace("{{ABSTRACTION}}", abstraction)


def apply_chat_template_text(tokenizer, user_content: str) -> str:
    messages = [{"role": "user", "content": user_content}]
    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return text.replace("<think>", "").replace("</think>", "").rstrip()


def extract_abstraction_from_tag(text: str) -> str | None:
    match = re.search(r"<abstraction>([\s\S]*?)</abstraction>", text)
    if match:
        extracted = match.group(1).strip()
        return extracted or None

    open_tag = "<abstraction>"
    start = text.find(open_tag)
    if start != -1:
        extracted = text[start + len(open_tag) :].strip()
        return extracted or None

    return None


def append_solution(prefix_text: str, solution_text: str) -> str:
    prefix = prefix_text.rstrip()
    solution = solution_text.strip()
    if not prefix:
        return solution
    return f"{prefix}\n{solution}"


def tokenize_ids(tokenizer, text: str) -> list[int]:
    return tokenizer(text, add_special_tokens=False).input_ids


async def post_json(url: str, payload: dict) -> dict:
    timeout = aiohttp.ClientTimeout(total=None)
    session = aiohttp.ClientSession(timeout=timeout)
    try:
        async with session.post(url, json=payload) as resp:
            text = await resp.text()
            resp.raise_for_status()
            return json.loads(text)
    finally:
        await session.close()


async def compute_ppl_via_vllm(
    reward_router_address: str,
    reward_model_name: str,
    reward_model_tokenizer,
    prefix_text: str,
    solution_text: str,
) -> float:
    full_text = append_solution(prefix_text, solution_text)
    prefix_ids = tokenize_ids(reward_model_tokenizer, prefix_text)
    full_ids = tokenize_ids(reward_model_tokenizer, full_text)

    payload = {
        "model": reward_model_name,
        "prompt": full_ids,
        "max_tokens": 1,
        "temperature": 0.0,
        "top_p": 1.0,
        "echo": True,
        "logprobs": 1,
        "return_tokens_as_token_ids": True,
    }
    result = await post_json(f"http://{reward_router_address}/v1/completions", payload)

    choice = result["choices"][0]
    token_logprobs = choice["logprobs"]["token_logprobs"]
    full_prompt_logprobs = token_logprobs[: len(full_ids)]
    solution_logprobs = [value for value in full_prompt_logprobs[len(prefix_ids) :] if value is not None]

    if not solution_logprobs:
        raise ValueError("No solution token logprobs were returned from the reward-model server.")

    mean_nll = -sum(solution_logprobs) / len(solution_logprobs)
    return math.exp(mean_nll)


async def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict,
    reward_router_address: str,
    reward_model_tokenizer,
    unconditional_lambda: float = 0.1,
    clip_reward: bool = True,
    positive_improvement_only: bool = True,
    hint_template_path: str = DEFAULT_HINT_TEMPLATE,
    reward_model_name: str = "Qwen/Qwen3-1.7B",
):
    del data_source, ground_truth

    abstraction = extract_abstraction_from_tag(solution_str or "")
    if abstraction is None or not abstraction.strip() or "\\boxed" in abstraction:
        return {
            "score": 0.0,
            "abstraction_valid": False,
            "ppl_improvement": 0.0,
            "num_solutions": 0,
        }

    problem = extra_info["problem"]
    correct_solutions = list(extra_info["correct_solutions"])
    baseline_ppls = list(extra_info["baseline_ppls"])
    unconditional_ppls = list(extra_info["unconditional_ppls"])
    template = load_template(hint_template_path)

    conditioned_prompt = render_template(template, problem=problem, abstraction=abstraction)
    conditioned_prefix = apply_chat_template_text(reward_model_tokenizer, conditioned_prompt)
    unconditional_prompt = render_template(template, problem="", abstraction=abstraction)
    unconditional_prefix = apply_chat_template_text(reward_model_tokenizer, unconditional_prompt)

    try:
        conditioned_tasks = [
            compute_ppl_via_vllm(
                reward_router_address=reward_router_address,
                reward_model_name=reward_model_name,
                reward_model_tokenizer=reward_model_tokenizer,
                prefix_text=conditioned_prefix,
                solution_text=solution,
            )
            for solution in correct_solutions
        ]
        unconditional_tasks = [
            compute_ppl_via_vllm(
                reward_router_address=reward_router_address,
                reward_model_name=reward_model_name,
                reward_model_tokenizer=reward_model_tokenizer,
                prefix_text=unconditional_prefix,
                solution_text=solution,
            )
            for solution in correct_solutions
        ]
        conditioned_values = await asyncio.gather(*conditioned_tasks)
        unconditional_values = await asyncio.gather(*unconditional_tasks)
    except Exception as exc:
        return {
            "score": 0.0,
            "abstraction_valid": True,
            "ppl_improvement": 0.0,
            "num_solutions": len(correct_solutions),
            "reward_error": str(exc),
        }

    per_solution_rewards: list[float] = []
    for baseline_ppl, conditioned_ppl, uncond_baseline, uncond_ppl in zip(
        baseline_ppls,
        conditioned_values,
        unconditional_ppls,
        unconditional_values,
        strict=True,
    ):
        baseline_improvement = (baseline_ppl - conditioned_ppl) / baseline_ppl
        unconditional_improvement = (uncond_baseline - uncond_ppl) / uncond_baseline
        if positive_improvement_only:
            baseline_improvement = max(0.0, baseline_improvement)
            unconditional_improvement = max(0.0, unconditional_improvement)
        per_solution_rewards.append(baseline_improvement - unconditional_lambda * unconditional_improvement)

    reward = sum(per_solution_rewards) / len(per_solution_rewards)
    if clip_reward:
        reward = max(0.0, reward)

    return {
        "score": reward,
        "abstraction_valid": True,
        "ppl_improvement": sum(per_solution_rewards) / len(per_solution_rewards),
        "mean_conditioned_ppl": sum(conditioned_values) / len(conditioned_values),
        "mean_unconditional_ppl": sum(unconditional_values) / len(unconditional_values),
        "num_solutions": len(correct_solutions),
    }
