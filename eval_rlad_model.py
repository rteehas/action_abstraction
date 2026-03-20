from __future__ import annotations

import argparse
import gc
import json
from functools import partial
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk
from process_deepscaler_dataset import compute_num_correct, get_generated_answers
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


DEFAULT_DATASET = "aime2025_amc2023_eval_set"
DEFAULT_ABSTRACTION_MODEL = "CMU-AIRe/RLAD-Hint-Gen"
DEFAULT_SOLVER_MODEL = "CMU-AIRe/RLAD-Sol-Gen"
DEFAULT_NUM_TRIALS = 4
DEFAULT_NUM_SOLVE_SAMPLES = 16
PROMPT_TEMPLATE_DIR = Path(
    "/scratch/rst306/action_abstractions/action_abstraction/prompt_templates"
)
ABSTRACTION_TEMPLATE_PATH = (
    PROMPT_TEMPLATE_DIR / "rlad_abstraction_generation_prompt_template.txt"
)
SOLVER_TEMPLATE_PATH = PROMPT_TEMPLATE_DIR / "rlad_solver_template.txt"


def load_prompt_template(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def make_abstraction_generation_prompt(problem: str, prompt_template: str) -> str:
    return prompt_template.replace("{problem_description}", problem)


def make_solver_prompt(
    problem: str,
    cheatsheet: str,
    prompt_template: str,
) -> str:
    return (
        prompt_template.replace("{cheatsheet}", cheatsheet).replace(
            "{problem_description}", problem
        )
    )


def maybe_apply_chat_template(
    tokenizer: AutoTokenizer,
    prompt: str,
    enable_thinking: bool,
) -> str:
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )



def generate_abstractions_for_trial(
    dataset,
    prompt_template: str,
    model_name: str,
    seed: int,
    num_gpus: int,
    temperature: float,
):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    llm = LLM(
        model=model_name,
        tensor_parallel_size=num_gpus,
        max_num_batched_tokens=8192,
    )
    sampling = SamplingParams(
        temperature=temperature,
        top_p=0.95,
        top_k=20,
        max_tokens=32768,
        seed=seed,
    )

    prompts = []
    for ex in tqdm(dataset, desc="creating abstraction messages"):
        prompt = make_abstraction_generation_prompt(ex["problem"], prompt_template)
        prompts.append(maybe_apply_chat_template(tokenizer, prompt, enable_thinking=False))
    print("Abstraction prompts")
    print(prompts[0])
    outputs = llm.generate(prompts, sampling)
    abstractions = [out.outputs[0].text for out in outputs]

    del outputs
    del llm
    gc.collect()
    torch.cuda.empty_cache()

    return abstractions


def generate_solutions_for_trial(
    dataset,
    prompt_template: str,
    model_name: str,
    seed: int,
    num_gpus: int,
    temperature: float,
    num_samples: int,
):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    llm = LLM(model=model_name, tensor_parallel_size=num_gpus)
    sampling = SamplingParams(
        temperature=temperature,
        top_p=0.95,
        top_k=20,
        max_tokens=32768,
        n=num_samples,
        seed=seed,
    )

    prompts = []
    for ex in tqdm(dataset, desc="creating solve messages"):
        prompt = make_solver_prompt(
            problem=ex["problem"],
            cheatsheet=ex["generated_abstraction"],
            prompt_template=prompt_template,
        )
        prompts.append(maybe_apply_chat_template(tokenizer, prompt, enable_thinking=True))
    print("Solve prompts")
    print(prompts[0])
    outputs = llm.generate(prompts, sampling)
    problem_solutions = [[candidate.text for candidate in out.outputs] for out in outputs]

    del outputs
    del llm
    gc.collect()
    torch.cuda.empty_cache()

    return problem_solutions


def metrics_from_scored(ds_scored):
    num_rows = ds_scored.num_rows
    total_samples = (
        int(sum(len(answers) for answers in ds_scored["generated_answer"]))
        if num_rows
        else 0
    )
    total_correct_samples = int(sum(ds_scored["num_correct"])) if num_rows else 0
    pass_at_1 = float(np.mean(ds_scored["passrate"])) if num_rows else 0.0
    pass_at_16 = (
        float(np.mean([num_correct > 0 for num_correct in ds_scored["num_correct"]]))
        if num_rows
        else 0.0
    )
    num_invalid = int(
        sum(
            1
            for answers in ds_scored["generated_answer"]
            for answer in answers
            if answer is None
        )
    )

    return {
        "num_rows": int(num_rows),
        "total_samples": total_samples,
        "total_correct_samples": total_correct_samples,
        "pass_at_1": pass_at_1,
        "pass_at_16": pass_at_16,
        "num_invalid": num_invalid,
    }


def per_source_report(ds_scored):
    if "source" not in ds_scored.column_names:
        return {}

    out = {}
    for source in sorted(set(ds_scored["source"])):
        ds_source = ds_scored.filter(lambda ex, source=source: ex["source"] == source)
        out[source] = metrics_from_scored(ds_source)
    return out


def summarize_reports(reports):
    def summarize_block(blocks):
        keys = [
            "num_rows",
            "total_samples",
            "total_correct_samples",
            "pass_at_1",
            "pass_at_16",
            "num_invalid",
        ]
        summary = {}
        for key in keys:
            values = np.array([block[key] for block in blocks], dtype=np.float64)
            summary[key] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            }
        return summary

    overall_summary = summarize_block([report["overall"] for report in reports])
    sources = sorted({source for report in reports for source in report["by_source"]})
    by_source_summary = {
        source: summarize_block([report["by_source"][source] for report in reports])
        for source in sources
    }

    return {
        "abstraction_model": reports[0]["abstraction_model"] if reports else "",
        "solver_model": reports[0]["solver_model"] if reports else "",
        "overall": overall_summary,
        "by_source": by_source_summary,
    }


def save_trial_outputs(
    output_dir: str | Path,
    trial_idx: int,
    ds_with_solutions,
    ds_scored,
    report: dict,
) -> None:
    trial_dir = Path(output_dir) / f"trial_{trial_idx:02d}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    ds_with_solutions.save_to_disk(str(trial_dir / "with_solutions"))
    ds_scored.save_to_disk(str(trial_dir / "scored"))
    (trial_dir / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )


def run_one_trial(
    trial_idx: int,
    base_seed: int,
    dataset,
    abstraction_prompt_template: str,
    solver_prompt_template: str,
    abstraction_model: str,
    solver_model: str,
    abstraction_temp: float,
    solver_temp: float,
    num_solve_samples: int,
    num_gpus: int,
):
    seed = base_seed + trial_idx

    abstraction_generations = generate_abstractions_for_trial(
        dataset=dataset,
        prompt_template=abstraction_prompt_template,
        model_name=abstraction_model,
        seed=seed,
        num_gpus=num_gpus,
        temperature=abstraction_temp,
    )

    ds_with_abstractions = dataset.add_column(
        "generated_abstraction", abstraction_generations
    )
    ds_filtered = ds_with_abstractions.filter(
        lambda ex: ex["generated_abstraction"] is not None
    )

    problem_solutions = generate_solutions_for_trial(
        dataset=ds_filtered,
        prompt_template=solver_prompt_template,
        model_name=solver_model,
        seed=seed,
        num_gpus=num_gpus,
        temperature=solver_temp,
        num_samples=num_solve_samples,
    )

    ds_with_solutions = ds_filtered.add_column(
        "abstraction_conditioned_problem_solutions", problem_solutions
    )
    get_answers = partial(
        get_generated_answers,
        answer_column="abstraction_conditioned_problem_solutions",
    )
    ds_scored = ds_with_solutions.map(get_answers).map(compute_num_correct)

    return ds_with_solutions, ds_scored, {
        "trial": trial_idx,
        "seed": seed,
        "abstraction_model": abstraction_model,
        "solver_model": solver_model,
        "overall": metrics_from_scored(ds_scored),
        "by_source": per_source_report(ds_scored),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET)
    parser.add_argument("--n_trials", type=int, default=DEFAULT_NUM_TRIALS)
    parser.add_argument("--base_seed", type=int, default=1234)
    parser.add_argument(
        "--abstraction_model",
        type=str,
        default=DEFAULT_ABSTRACTION_MODEL,
    )
    parser.add_argument("--solver_model", type=str, default=DEFAULT_SOLVER_MODEL)
    parser.add_argument("--abstraction_temp", type=float, default=1.0)
    parser.add_argument("--solver_temp", type=float, default=0.6)
    parser.add_argument(
        "--num_solve_samples",
        type=int,
        default=DEFAULT_NUM_SOLVE_SAMPLES,
    )
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--save_model_outputs", action="store_true")
    parser.add_argument("--output_dir", type=str, default="")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.num_gpus < 1:
        raise ValueError("--num_gpus must be at least 1")
    if args.num_solve_samples < 1:
        raise ValueError("--num_solve_samples must be at least 1")
    if args.save_model_outputs and not args.output_dir:
        raise ValueError("--output_dir must be set when --save_model_outputs is used")

    print(f"Device name = {torch.cuda.get_device_name(0)}")

    abstraction_prompt_template = load_prompt_template(ABSTRACTION_TEMPLATE_PATH)
    solver_prompt_template = load_prompt_template(SOLVER_TEMPLATE_PATH)
    dataset = load_from_disk(args.dataset)

    reports = []
    for trial_idx in range(args.n_trials):
        torch.manual_seed(args.base_seed + trial_idx)
        ds_with_solutions, ds_scored, report = run_one_trial(
            trial_idx=trial_idx,
            base_seed=args.base_seed,
            dataset=dataset,
            abstraction_prompt_template=abstraction_prompt_template,
            solver_prompt_template=solver_prompt_template,
            abstraction_model=args.abstraction_model,
            solver_model=args.solver_model,
            abstraction_temp=args.abstraction_temp,
            solver_temp=args.solver_temp,
            num_solve_samples=args.num_solve_samples,
            num_gpus=args.num_gpus,
        )
        reports.append(report)
        print(f"TRIAL REPORT {trial_idx}: {report}")

        if args.save_model_outputs:
            save_trial_outputs(
                output_dir=args.output_dir,
                trial_idx=trial_idx,
                ds_with_solutions=ds_with_solutions,
                ds_scored=ds_scored,
                report=report,
            )

    summary = summarize_reports(reports)
    print("\nALL TRIAL REPORTS")
    for report in reports:
        print(report)

    print("\nMEAN/STD SUMMARY")
    print(summary)

    if args.save_model_outputs:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "reports.json").write_text(
            json.dumps(reports, indent=2), encoding="utf-8"
        )
        (out_dir / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
