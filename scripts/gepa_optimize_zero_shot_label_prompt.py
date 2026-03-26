from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import textwrap
from pathlib import Path
from typing import Any

from gepa.optimize_anything import GEPAConfig, EngineConfig, ReflectionConfig, optimize_anything

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--seed_prompt_path', type=str, required=True)
    p.add_argument('--dataset_path', type=str, required=True)
    p.add_argument('--solver_prompt_path', type=str, required=True)
    p.add_argument('--output_root', type=str, required=True)
    p.add_argument('--max_rows', type=int, default=26)
    p.add_argument('--reflection_model', type=str, default='openai/gpt-5-nano')
    p.add_argument('--max_metric_calls', type=int, default=8)
    p.add_argument('--max_model_len', type=int, default=32768)
    p.add_argument('--max_num_batched_tokens', type=int, default=4096)
    p.add_argument('--gpu_memory_utilization', type=float, default=0.9)
    p.add_argument('--solver_report_title', type=str, default='GEPA Zero-Shot Prompt Eval')
    return p.parse_args()


def repo_path(s: str) -> Path:
    p = Path(s)
    if p.is_absolute():
        return p
    return REPO_ROOT / p


def candidate_hash(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def summarize_rows(rows_path: Path, limit: int = 5) -> list[dict[str, Any]]:
    rows = json.loads(rows_path.read_text())
    failures = [r for r in rows if not r.get('solver_correct')]
    summaries = []
    for row in failures[:limit]:
        summaries.append(
            {
                'row_id': row.get('row_id'),
                'expected_answer': row.get('answer'),
                'predicted_answer': row.get('solver_predicted_answer'),
                'problem_excerpt': (row.get('problem') or '')[:240],
                'abstraction_excerpt': (row.get('abstraction') or '')[:240],
            }
        )
    return summaries


def build_feedback(run_dir: Path, rescored: dict[str, Any]) -> dict[str, Any]:
    feedback: dict[str, Any] = {
        'scores': {'rescored_accuracy': rescored['accuracy']},
        'Summary': f"rescored {rescored['num_correct']}/{rescored['num_rows']} = {rescored['accuracy']:.4f}",
        'AbsoluteLift': rescored.get('absolute_accuracy_lift'),
        'RunDir': str(run_dir),
    }
    rows_path = run_dir / 'rows.json'
    if rows_path.exists():
        feedback['FailureCases'] = summarize_rows(rows_path)
    report_path = run_dir / 'report.json'
    if report_path.exists():
        raw = read_json(report_path)
        feedback['RawMetric'] = raw.get('accuracy')
    return feedback


def validate_candidate(candidate: str) -> tuple[bool, str]:
    required_placeholders = ['{{PROBLEM}}', '{{CORRECT_SOLUTIONS_BLOCK}}', '{{INCORRECT_SOLUTIONS_BLOCK}}']
    missing = [placeholder for placeholder in required_placeholders if placeholder not in candidate]
    if missing:
        return False, f'missing placeholders: {missing}'
    if candidate.count('<abstraction>') != 1 or candidate.count('</abstraction>') != 1:
        return False, 'candidate must contain exactly one literal <abstraction>...</abstraction> instruction block'
    if candidate.count('Abstraction:') != 1:
        return False, 'candidate must contain exactly one final Abstraction: marker'
    suffix = candidate.split('Abstraction:', 1)[1].strip()
    if suffix:
        return False, 'candidate must not include a pre-filled abstraction after the final Abstraction: marker'
    return True, ''


def run_eval(candidate: str, example: Any = None, opt_state: Any = None) -> tuple[float, dict[str, Any]]:
    del example, opt_state
    args = run_eval.args
    valid, reason = validate_candidate(candidate)
    if not valid:
        side_info = {
            'scores': {'rescored_accuracy': 0.0},
            'Summary': f'invalid candidate: {reason}',
            'InvalidCandidate': True,
        }
        return 0.0, side_info
    out_root = repo_path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    h = candidate_hash(candidate)
    prompt_path = out_root / f'candidate_{h}.txt'
    run_dir = out_root / f'run_{h}'
    prompt_path.write_text(candidate)

    rescored_path = run_dir / 'rescored_report.json'
    if not rescored_path.exists():
        eval_cmd = [
            '/workspace/miniconda3/bin/conda', 'run', '--no-capture-output', '-n', 'abstraction', 'python',
            'scripts/eval_contrastive_abstractions.py',
            '--dataset_path', args.dataset_path,
            '--split', 'test',
            '--max_rows', str(args.max_rows),
            '--report_title', f'{args.solver_report_title}: {h}',
            '--abstraction_prompt_path', str(prompt_path),
            '--few_shots_path', '',
            '--solver_prompt_path', args.solver_prompt_path,
            '--output_dir', str(run_dir),
            '--max_model_len', str(args.max_model_len),
            '--max_num_batched_tokens', str(args.max_num_batched_tokens),
            '--gpu_memory_utilization', str(args.gpu_memory_utilization),
        ]
        subprocess.run(eval_cmd, cwd=REPO_ROOT, check=True)
        rescore_cmd = [
            '/workspace/miniconda3/bin/conda', 'run', '--no-capture-output', '-n', 'abstraction', 'python',
            'scripts/rescore_contrastive_eval.py', '--run_dir', str(run_dir),
        ]
        subprocess.run(rescore_cmd, cwd=REPO_ROOT, check=True)

    rescored = read_json(rescored_path)
    score = float(rescored['accuracy'])
    side_info = build_feedback(run_dir, rescored)
    side_info['Output'] = {'run_dir': str(run_dir), 'score': score}
    return score, side_info


def main() -> None:
    args = parse_args()
    run_eval.args = args
    seed_prompt = repo_path(args.seed_prompt_path).read_text()
    output_root = repo_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if not os.environ.get('OPENAI_API_KEY'):
        raise RuntimeError('OPENAI_API_KEY must be set for GEPA reflection.')

    objective = (
        'Maximize rescored downstream solve accuracy on the fixed 26-row contrastive abstraction subset by improving the zero-shot abstraction-labeling prompt. '
        'The candidate is the full prompt text. Higher rescored accuracy is better.'
    )
    background = textwrap.dedent(
        f'''\
        Domain: zero-shot contrastive abstraction labeling for downstream math problem solving.
        Fixed evaluation: dataset={args.dataset_path}, max_rows={args.max_rows}, solver prompt={args.solver_prompt_path}.
        Current best zero-shot seed prompt is {args.seed_prompt_path}.
        Known strong behavior: simpler teacher-note framing and explicit structural moves help.
        Known weak behavior: heavy rubric language, rigid sentence templates, generic filler, and example leakage into the template hurt.
        Keep outputs as a single prompt template with placeholders {{PROBLEM}}, {{CORRECT_SOLUTIONS_BLOCK}}, and {{INCORRECT_SOLUTIONS_BLOCK}}.
        Do not append any example abstraction, filled-in answer, or problem-specific text after the final Abstraction: marker.
        Do not add extra <abstraction> blocks beyond the single literal format instruction in the template.
        '''
    ).strip()

    config = GEPAConfig(
        engine=EngineConfig(
            run_dir=str(output_root / 'gepa_run'),
            max_metric_calls=args.max_metric_calls,
            display_progress_bar=False,
            cache_evaluation=True,
            cache_evaluation_storage='disk',
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
        'best_idx': best_idx,
        'best_score': result.val_aggregate_scores[best_idx],
        'num_candidates': result.num_candidates,
        'total_metric_calls': result.total_metric_calls,
        'run_dir': result.run_dir,
        'best_candidate': result.best_candidate,
        'all_scores': result.val_aggregate_scores,
    }
    (output_root / 'gepa_summary.json').write_text(json.dumps(summary, indent=2))
    (output_root / 'gepa_best_prompt.txt').write_text(result.best_candidate)
    (output_root / 'gepa_result.json').write_text(json.dumps(result.to_dict(), indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
