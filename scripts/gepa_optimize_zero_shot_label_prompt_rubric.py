from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import textwrap
import time
from pathlib import Path
from typing import Any

from datasets import load_from_disk
from openai import APIConnectionError, APIError, APITimeoutError, OpenAI, RateLimitError
from gepa.optimize_anything import GEPAConfig, EngineConfig, ReflectionConfig, optimize_anything

REPO_ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = {
    "structural_centrality": 0.28,
    "executability": 0.25,
    "transferability": 0.15,
    "problem_grounded_inferability": 0.22,
    "faithfulness_and_relevance": 0.07,
    "pitfall_checkpoint_value": 0.03,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--seed_prompt_path", type=str, required=True)
    p.add_argument("--dataset_path", type=str, required=True)
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--max_rows", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rubric_path", type=str, default="outputs/2026-03-18/contrastive_abstraction_prompting/abstraction_quality_rubric.md")
    p.add_argument("--output_root", type=str, required=True)
    p.add_argument("--reflection_model", type=str, default="gpt-5-mini")
    p.add_argument("--label_model", type=str, default="gpt-5-mini")
    p.add_argument("--judge_model", type=str, default="gpt-5-mini")
    p.add_argument("--max_metric_calls", type=int, default=8)
    p.add_argument("--label_temperature", type=float, default=None)
    p.add_argument("--judge_temperature", type=float, default=None)
    return p.parse_args()


def repo_path(s: str) -> Path:
    p = Path(s)
    if p.is_absolute():
        return p
    return REPO_ROOT / p


def candidate_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def validate_candidate(candidate: str) -> tuple[bool, str]:
    required_placeholders = ["{{PROBLEM}}", "{{CORRECT_SOLUTIONS_BLOCK}}", "{{INCORRECT_SOLUTIONS_BLOCK}}"]
    missing = [placeholder for placeholder in required_placeholders if placeholder not in candidate]
    if missing:
        return False, f"missing placeholders: {missing}"
    if candidate.count("<abstraction>") != 1 or candidate.count("</abstraction>") != 1:
        return False, "candidate must contain exactly one literal <abstraction>...</abstraction> instruction block"
    if candidate.count("Abstraction:") != 1:
        return False, "candidate must contain exactly one final Abstraction: marker"
    suffix = candidate.split("Abstraction:", 1)[1].strip()
    if suffix:
        return False, "candidate must not include a pre-filled abstraction after the final Abstraction: marker"
    return True, ""


def format_solution_block(solutions: list[str]) -> str:
    parts = []
    for i, sol in enumerate(solutions, start=1):
        parts.append(f"Solution {i}:\n{sol.strip()}")
    return "\n\n".join(parts)


def render_prompt(template: str, row: dict[str, Any]) -> str:
    return (
        template.replace("{{PROBLEM}}", row["problem"])
        .replace("{{CORRECT_SOLUTIONS_BLOCK}}", format_solution_block(row["selected_correct_solutions"]))
        .replace("{{INCORRECT_SOLUTIONS_BLOCK}}", format_solution_block(row["selected_incorrect_solutions"]))
    )


def extract_abstraction(text: str) -> str:
    m = re.search(r"<abstraction>(.*?)</abstraction>", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return text.strip()


def get_text_content(resp: Any) -> str:
    if hasattr(resp, "choices"):
        return resp.choices[0].message.content or ""
    return getattr(resp, "output_text", "") or ""


def _chat_create(client: OpenAI, **kwargs: Any) -> Any:
    errors = (APITimeoutError, APIConnectionError, RateLimitError, APIError)
    for attempt in range(5):
        try:
            return client.chat.completions.create(timeout=120, **kwargs)
        except errors:
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


def chat_text(client: OpenAI, model: str, system: str, user: str, temperature: float | None) -> str:
    kwargs = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    if temperature is not None:
        kwargs["temperature"] = temperature
    resp = _chat_create(client, **kwargs)
    return get_text_content(resp)


def chat_json(client: OpenAI, model: str, system: str, user: str, temperature: float | None) -> dict[str, Any]:
    kwargs = {
        "model": model,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    if temperature is not None:
        kwargs["temperature"] = temperature
    resp = _chat_create(client, **kwargs)
    text = get_text_content(resp)
    return json.loads(text)


def judge_abstraction(client: OpenAI, judge_model: str, rubric_text: str, row: dict[str, Any], abstraction: str, temperature: float) -> dict[str, Any]:
    abstraction = abstraction.strip()
    if not abstraction:
        zero_dims = {key: 0 for key in WEIGHTS}
        return {
            "dimensions": zero_dims,
            "weighted_score": 0.0,
            "summary": "Empty abstraction.",
            "strongest_issue": "Empty output.",
        }

    system = (
        "You are a strict and somewhat harsh evaluator of math-solving abstractions. "
        "Score the abstraction only against the rubric and the provided good/bad traces. "
        "First identify the decisive crux from the good trace(s), then judge separately whether the abstraction says how to use that crux on this problem, whether it states when the move applies, and whether a problem-only student could plausibly infer it from visible cues. "
        "Use the full scale. Reserve 4 for exceptional cases. A useful but imperfect abstraction should usually receive 3 on strong dimensions, not 4. "
        "Deduct for generic wording, mild overreach, unnecessary detail, canned filler, tutorialized background, or partial mismatch with the good traces. "
        "A broad principle name without the decisive concrete move or local instantiation should not score above 2 on structural_centrality or executability. "
        "A slogan or topic label without a real move should not score above 1 on structural_centrality or executability. "
        "Any explicit final value, answer-adjacent conclusion, or trace-only fact not plausibly visible from the problem should not score above 1 on problem_grounded_inferability or transferability. "
        "A mini-solution that is overly tied to this exact instance should not score above 1 on transferability. "
        "Unsupported machinery should not score above 1 on faithfulness_and_relevance. "
        "If more than about half the note is background explanation, textbook method talk, or illustrative generality rather than the decisive move and its local use, faithfulness_and_relevance should not score above 2. "
        "If no pitfall or checkpoint is genuinely needed, treat pitfall_checkpoint_value = 2 as neutral rather than penalizing the abstraction for omitting one. "
        "Return valid JSON only."
    )
    user = textwrap.dedent(
        f"""\
        Rubric:
        {rubric_text}

        Problem:
        {row['problem']}

        Correct solution trace(s):
        {format_solution_block(row['selected_correct_solutions'])}

        Incorrect solution trace(s):
        {format_solution_block(row['selected_incorrect_solutions'])}

        Candidate abstraction:
        {abstraction}

        Apply these rules before scoring:
        - Identify the decisive crux from the good trace(s).
        - Ask separately whether the abstraction says how to use that crux on this problem.
        - If the abstraction names a broad principle but misses the decisive move or its local instantiation, cap structural_centrality and executability at 2.
        - If the abstraction is just a topic label or slogan, cap structural_centrality and executability at 1.
        - If the abstraction includes a final value, answer-adjacent conclusion, or trace-only fact not plausibly visible from the problem, cap problem_grounded_inferability and transferability at 1.
        - If the abstraction is a mini-solution for this exact instance, cap transferability at 1.
        - If the abstraction adds unsupported machinery not grounded in the good trace(s), cap faithfulness_and_relevance at 1.
        - If more than about half the note is background explanation, textbook method talk, or illustrative generality rather than the decisive move and its local use, cap faithfulness_and_relevance at 2.
        - Absence of a pitfall is neutral when no pitfall is needed.
        - Reward named principles only when they clarify the decisive move and its trigger.
        - Reward notes that tell the solver what visible cue in the problem should trigger the move.
        - Reward representation shifts that reconceptualize the problem in a way the solver can act on.

        Return a JSON object with exactly these keys:
        - structural_centrality: integer 0-4
        - executability: integer 0-4
        - transferability: integer 0-4
        - problem_grounded_inferability: integer 0-4
        - faithfulness_and_relevance: integer 0-4
        - pitfall_checkpoint_value: integer 0-4
        - summary: short string
        - strongest_issue: short string
        """
    )
    obj = chat_json(client, judge_model, system, user, temperature)
    dims = {}
    for key in WEIGHTS:
        val = int(obj[key])
        if not 0 <= val <= 4:
            raise ValueError(f"judge returned invalid score {key}={val}")
        dims[key] = val
    weighted = sum((dims[k] / 4.0) * WEIGHTS[k] for k in WEIGHTS)
    return {
        "dimensions": dims,
        "weighted_score": weighted,
        "summary": str(obj.get("summary", "")),
        "strongest_issue": str(obj.get("strongest_issue", "")),
    }


def build_feedback(per_row: list[dict[str, Any]], mean_score: float, run_dir: Path) -> dict[str, Any]:
    worst = sorted(per_row, key=lambda x: x["weighted_score"])[:5]
    return {
        "scores": {"rubric_score": mean_score},
        "Summary": f"mean rubric score = {mean_score:.4f} across {len(per_row)} rows",
        "RunDir": str(run_dir),
        "WorstRows": [
            {
                "row_id": row["row_id"],
                "weighted_score": row["weighted_score"],
                "summary": row["judge_summary"],
                "strongest_issue": row["strongest_issue"],
                "abstraction_excerpt": row["abstraction"][:240],
            }
            for row in worst
        ],
    }


def load_eval_rows(dataset_path: Path, split: str, max_rows: int, seed: int) -> list[dict[str, Any]]:
    ds = load_from_disk(str(dataset_path))[split]
    indices = list(range(len(ds)))
    random.Random(seed).shuffle(indices)
    chosen = indices[:max_rows]
    return [ds[i] for i in chosen]


def run_eval(candidate: str, example: Any = None, opt_state: Any = None) -> tuple[float, dict[str, Any]]:
    del example, opt_state
    args = run_eval.args
    valid, reason = validate_candidate(candidate)
    if not valid:
        return 0.0, {
            "scores": {"rubric_score": 0.0},
            "Summary": f"invalid candidate: {reason}",
            "InvalidCandidate": True,
        }

    out_root = repo_path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    h = candidate_hash(candidate)
    prompt_path = out_root / f"candidate_{h}.txt"
    run_dir = out_root / f"run_{h}"
    run_dir.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text(candidate)

    per_row_path = run_dir / "rubric_rows.json"
    score_path = run_dir / "rubric_report.json"
    if not score_path.exists():
        rows_out = []
        for row in run_eval.eval_rows:
            rendered = render_prompt(candidate, row)
            raw_output = chat_text(
                run_eval.client,
                args.label_model,
                system="Follow the user's prompt exactly and return only the requested abstraction.",
                user=rendered,
                temperature=args.label_temperature,
            )
            abstraction = extract_abstraction(raw_output)
            judge = judge_abstraction(
                run_eval.client,
                args.judge_model,
                run_eval.rubric_text,
                row,
                abstraction,
                args.judge_temperature,
            )
            rows_out.append(
                {
                    "row_id": row["row_id"],
                    "problem": row["problem"],
                    "abstraction": abstraction,
                    "raw_output": raw_output,
                    "weighted_score": judge["weighted_score"],
                    "judge_summary": judge["summary"],
                    "strongest_issue": judge["strongest_issue"],
                    **judge["dimensions"],
                }
            )
        mean_score = sum(r["weighted_score"] for r in rows_out) / len(rows_out)
        per_row_path.write_text(json.dumps(rows_out, indent=2))
        score_path.write_text(json.dumps({
            "num_rows": len(rows_out),
            "mean_rubric_score": mean_score,
            "label_model": args.label_model,
            "judge_model": args.judge_model,
            "split": args.split,
            "max_rows": args.max_rows,
            "seed": args.seed,
        }, indent=2))

    rows_out = json.loads(per_row_path.read_text())
    score_obj = json.loads(score_path.read_text())
    score = float(score_obj["mean_rubric_score"])
    side_info = build_feedback(rows_out, score, run_dir)
    side_info["Output"] = {"run_dir": str(run_dir), "score": score}
    return score, side_info


def main() -> None:
    args = parse_args()
    run_eval.args = args
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY must be set for GEPA reflection and rubric judging.")
    run_eval.client = OpenAI()
    run_eval.rubric_text = repo_path(args.rubric_path).read_text()
    run_eval.eval_rows = load_eval_rows(repo_path(args.dataset_path), args.split, args.max_rows, args.seed)

    output_root = repo_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    seed_prompt = repo_path(args.seed_prompt_path).read_text()

    objective = (
        "Maximize the average abstraction-quality rubric score on a fixed contrastive math subset. "
        "The score should reward structural centrality, executability, transferability, problem-grounded inferability, faithfulness, and pitfall value."
    )
    background = textwrap.dedent(
        f"""\
        Domain: zero-shot abstraction labeling for downstream math problem solving.
        Candidate: the full prompt template used to generate one abstraction from a problem plus good/bad traces.
        Label model: {args.label_model}.
        Judge model: {args.judge_model}.
        Reflection model: {args.reflection_model}.
        Fixed eval subset: dataset={args.dataset_path}, split={args.split}, rows={args.max_rows}, seed={args.seed}.
        The rubric rewards abstractions that name the decisive move, explain why a solver should notice it from cues in the problem, and stay grounded enough for a problem-only student to plausibly learn.
        Keep outputs as a single prompt template with placeholders {{PROBLEM}}, {{CORRECT_SOLUTIONS_BLOCK}}, and {{INCORRECT_SOLUTIONS_BLOCK}}.
        Do not append any example abstraction, filled-in answer, or problem-specific text after the final Abstraction: marker.
        Do not add extra <abstraction> blocks beyond the single literal format instruction in the template.
        High-scoring prompts usually elicit: the decisive structural move, one genuinely central execution cue when needed, and a useful pitfall or checkpoint.
        Low-scoring prompts usually elicit: generic slogans, forced parameterization, irrelevant filler, or disguised worked solutions.
        """
    ).strip()

    config = GEPAConfig(
        engine=EngineConfig(
            run_dir=str(output_root / "gepa_run"),
            max_metric_calls=args.max_metric_calls,
            display_progress_bar=False,
            cache_evaluation=True,
            cache_evaluation_storage="disk",
            best_example_evals_k=5,
        ),
        reflection=ReflectionConfig(
            reflection_lm=args.reflection_model,
            skip_perfect_score=False,
        ),
        refiner=None,
        merge=None,
    )

    result = optimize_anything(
        seed_candidate=seed_prompt,
        evaluator=run_eval,
        dataset=None,
        valset=None,
        objective=objective,
        background=background,
        config=config,
    )

    best_idx = result.best_idx
    summary = {
        "best_idx": best_idx,
        "best_score": result.val_aggregate_scores[best_idx],
        "num_candidates": result.num_candidates,
        "total_metric_calls": result.total_metric_calls,
        "run_dir": result.run_dir,
        "best_candidate": result.best_candidate,
        "all_scores": result.val_aggregate_scores,
        "label_model": args.label_model,
        "judge_model": args.judge_model,
        "reflection_model": args.reflection_model,
        "split": args.split,
        "max_rows": args.max_rows,
        "seed": args.seed,
    }
    (output_root / "gepa_summary.json").write_text(json.dumps(summary, indent=2))
    (output_root / "gepa_best_prompt.txt").write_text(result.best_candidate)
    (output_root / "gepa_result.json").write_text(json.dumps(result.to_dict(), indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
