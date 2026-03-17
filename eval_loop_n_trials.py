from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from datasets import load_dataset, load_from_disk
from tqdm import tqdm
from problem_solving import make_problem_solving_prompt
from functools import partial
from process_deepscaler_dataset import get_generated_answers, compute_num_correct
import gc
import torch
import numpy as np
import argparse
import json
from pathlib import Path


def make_problem_solving_prompt_one_shot(problem, abstraction, one_shot_problem, one_shot_abstraction, one_shot_solution):
    with open("/scratch/rst306/action_abstractions/action_abstraction/prompt_templates/abstraction_conditioned_problem_solving.txt", 'r') as fp:
        lines = fp.readlines()
    
    problem_solving_prompt_template = "".join(lines)

    prompt = problem_solving_prompt_template.replace("{{PROBLEM}}", problem).replace("{{ABSTRACTION}}", abstraction).replace("{{FEW_SHOT_PROBLEM}}", one_shot_problem).replace("{{FEW_SHOT_ABSTRACTION}}", one_shot_abstraction).replace("{{SOLUTION}}", one_shot_solution)
    return prompt

def make_problem_solving_prompt_hint_conditioned(problem, abstraction):
    with open("/scratch/rst306/action_abstractions/action_abstraction/prompt_templates/hint_conditioned_problem_solving.txt", 'r') as fp:
        lines = fp.readlines()
    
    problem_solving_prompt_template = "".join(lines)

    prompt = problem_solving_prompt_template.replace("{{PROBLEM}}", problem).replace("{{ABSTRACTION}}", abstraction)
    return prompt


def get_one_shot_example():
    from functools import partial
    from process_deepscaler_dataset import get_generated_answers, compute_num_correct

    dataset = load_from_disk("deepscaler_solutions_conditioned_on_prelabeled_abstractions")
    compute_corr = partial(compute_num_correct, prefix="abs_conditioned")
    get_ans = partial(get_generated_answers, answer_column="abstraction_conditioned_problem_solutions")
    dataset = dataset.map(get_ans).map(compute_corr)
    candidates = dataset.filter(lambda ex: ex["passrate"] == 0.25 and ex["abs_conditioned_passrate"] == 1.0)
    return candidates[0]["problem"], candidates[0]["abstraction"], candidates[0]["abstraction_conditioned_problem_solutions"][0]

def run_one_trial(
    trial_idx: int,
    base_seed: int,
    dataset,
    sft_prompt_template: str,
    base_model: str,
    sft_lora_path: str,
    solver_lora_path: str,
    solver_temp: float,
    one_shot: bool,
    save_model_outputs: bool = False,
    output_dir: str = "",
):
    seed = base_seed + trial_idx

    def make_abstration_generation_prompt(ex):
        problem = ex["problem"]
        prompt = sft_prompt_template.replace("{{PROBLEM}}", problem)
        ex["prompt"] = prompt
        return ex

    ds = dataset.map(make_abstration_generation_prompt)

    tokenizer = AutoTokenizer.from_pretrained(base_model)

    abstraction_sampling = SamplingParams(
        temperature=1.0, top_p=0.95, top_k=20, max_tokens=32768, seed=seed
    )

    if sft_lora_path == "":
        llm = LLM(
            model=base_model,
            max_num_batched_tokens=8192,
        )
    else:
        llm = LLM(
            model=base_model,
            enable_lora=True,
            max_num_batched_tokens=8192,
        )

        lora_req = LoRARequest(
            "abstraction-generation-warmstart",
            1,
            sft_lora_path,
        )

    abstraction_texts = []
    for ex in tqdm(ds, desc=f"[trial {trial_idx}] creating abstraction messages"):
        messages = [{"role": "user", "content": ex["prompt"]}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        abstraction_texts.append(text)

    if sft_lora_path == "":
        outputs = llm.generate(
            abstraction_texts,
            abstraction_sampling,
        )
    else:
        outputs = llm.generate(
            abstraction_texts,
            abstraction_sampling,
            lora_request=lora_req,
        )

    abstraction_generations = [out.outputs[0].text for out in outputs]

    ds_with_generations = ds.add_column("generated_abstraction", abstraction_generations)
    abstraction_column = "generated_abstraction"
    ds_filtered = ds_with_generations.filter(lambda ex: ex[abstraction_column] is not None)

    del outputs
    del llm
    gc.collect()
    torch.cuda.empty_cache()

    solve_sampling = SamplingParams(
        temperature=solver_temp, top_p=0.95, top_k=20, max_tokens=32768, n=1, seed=seed
    )

    if solver_lora_path == "":
        llm = LLM(model=base_model)
    else:
        llm = LLM(
            model=base_model,
            enable_lora=True,
        )

        solver_lora_req = LoRARequest(
            "solver-grpo",
            1,
            solver_lora_path,
        )

    solve_texts = []
    for ex in tqdm(ds_filtered, desc=f"[trial {trial_idx}] creating solve messages"):
        problem = ex["problem"]
        abstraction = ex[abstraction_column]
        if one_shot:
            one_shot_problem, one_shot_abstraction, one_shot_solution = get_one_shot_example()
            prompt = make_problem_solving_prompt_one_shot(problem, abstraction, one_shot_problem, one_shot_abstraction, one_shot_solution)
        else:
            if solver_lora_path == "":
                prompt = make_problem_solving_prompt(problem, abstraction)
            else:
                prompt = make_problem_solving_prompt_hint_conditioned(problem, abstraction)

        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        solve_texts.append(text)

    print("SOLVER TEXT")
    print(solve_texts[0])

    if solver_lora_path == "":
        outputs = llm.generate(solve_texts, solve_sampling)
    else:
        outputs = llm.generate(solve_texts, solve_sampling, lora_request=solver_lora_req)

    problem_solutions = []
    for out in outputs:
        problem_solutions.append([p.text for p in out.outputs])

    ds_with_solutions = ds_filtered.add_column(
        "abstraction_conditioned_problem_solutions", problem_solutions
    )

    get_ans = partial(
        get_generated_answers,
        answer_column="abstraction_conditioned_problem_solutions",
    )
    ds_scored = ds_with_solutions.map(get_ans).map(compute_num_correct)

    overall = metrics_from_scored(ds_scored)
    by_source = per_source_report(ds_scored)

    final_report = {
        "trial": trial_idx,
        "seed": seed,
        "abstraction_generator_lora_path": sft_lora_path,
        "solver_lora_path": solver_lora_path,
        "overall": overall,
        "by_source": by_source,
    }

    if save_model_outputs and output_dir:
        trial_dir = Path(output_dir) / f"trial_{trial_idx:02d}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        ds_with_solutions.save_to_disk(str(trial_dir / "with_solutions"))
        ds_scored.save_to_disk(str(trial_dir / "scored"))
        (trial_dir / "report.json").write_text(
            json.dumps(final_report, indent=2), encoding="utf-8"
        )
        print(f"[trial {trial_idx}] saved model outputs to {trial_dir}")

    del outputs
    del llm
    gc.collect()
    torch.cuda.empty_cache()

    return final_report

def metrics_from_scored(ds_scored):
    num_rows = ds_scored.num_rows
    num_correct = int(sum(ds_scored["passrate"]))
    acc = float(num_correct / num_rows) if num_rows else 0.0
    num_invalid = int(ds_scored.filter(lambda ex: ex["generated_answer"][0] is None).num_rows)
    return {
        "num_rows": int(num_rows),
        "num_correct": num_correct,
        "accuracy": acc,
        "num_invalid": num_invalid,
    }

def per_source_report(ds_scored):
    sources = sorted(set(ds_scored["source"]))
    out = {}
    for s in sources:
        ds_s = ds_scored.filter(lambda ex, s=s: ex["source"] == s)
        out[s] = metrics_from_scored(ds_s)
    return out

def summarize_reports(reports):
    def summarize_block(blocks):
        keys = ["num_rows", "num_correct", "accuracy", "num_invalid"]
        summary = {}
        for k in keys:
            vals = np.array([b[k] for b in blocks], dtype=np.float64)
            summary[k] = {
                "mean": float(vals.mean()),
                "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
            }
        return summary

    overall_summary = summarize_block([r["overall"] for r in reports])

    sources = sorted({s for r in reports for s in r["by_source"].keys()})
    per_source_summary = {
        s: summarize_block([r["by_source"][s] for r in reports])
        for s in sources
    }

    return {
        "abstraction_generator_lora_path": reports[0]["abstraction_generator_lora_path"] if reports else "",
        "solver_lora_path": reports[0]["solver_lora_path"] if reports else "",
        "overall": overall_summary,
        "by_source": per_source_summary,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_trials", type=int, default=5)
    parser.add_argument("--base_seed", type=int, default=1234)
    parser.add_argument("--sft_lora_path", type=str, default="")
    parser.add_argument("--solver_lora_path", type=str, default="")
    parser.add_argument("--solver_temp", type=float, default=0.6)
    parser.add_argument("--one_shot", action="store_true")
    parser.add_argument("--save_model_outputs", action="store_true")
    parser.add_argument("--output_dir", type=str, default="")
    args = parser.parse_args()

    if args.save_model_outputs and not args.output_dir:
        raise ValueError("--output_dir must be set when --save_model_outputs is used")

    print(f"Device name = {torch.cuda.get_device_name(0)}")

    with open(
        "/scratch/rst306/action_abstractions/action_abstraction/prompt_templates/sft_abstraction_generation.txt",
        "r",
    ) as fp:
        sft_prompt_template = fp.read()

    base_model = "Qwen/Qwen3-1.7B"
    sft_lora_path = args.sft_lora_path
    print(f"Evaluating lora path: {sft_lora_path} for abs gen")
    print(f"Evaluating lora path: {args.solver_lora_path} for solver")

    print("AIME 2025 and AMC 2023")
    dataset = load_from_disk("aime2025_amc2023_eval_set")

    reports = []
    for t in range(args.n_trials):
        torch.manual_seed(args.base_seed + t)
        report = run_one_trial(
            trial_idx=t,
            base_seed=args.base_seed,
            dataset=dataset,
            sft_prompt_template=sft_prompt_template,
            base_model=base_model,
            sft_lora_path=sft_lora_path,
            solver_lora_path=args.solver_lora_path,
            solver_temp=args.solver_temp,
            one_shot=args.one_shot,
            save_model_outputs=args.save_model_outputs,
            output_dir=args.output_dir,
        )
        reports.append(report)
        print(f"TRIAL REPORT {t}: {report}")

    summary = summarize_reports(reports)
    print("\nALL TRIAL REPORTS")
    for r in reports:
        print(r)

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
        print(f"Saved aggregate reports to {out_dir}")