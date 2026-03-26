from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--prompt_paths', nargs='+', required=True)
    parser.add_argument('--solver_prompt_path', type=str, default='prompt_templates/hint_conditioned_problem_solving.txt')
    parser.add_argument('--few_shots_path', type=str, default='prompt_templates/contrastive_abstraction_labeling_few_shots.txt')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--max_rows', type=int, default=26)
    parser.add_argument('--label_mode', choices=['contrastive', 'problem_only'], default='contrastive')
    parser.add_argument('--base_model', type=str, default='Qwen/Qwen3-1.7B')
    parser.add_argument('--output_root', type=str, required=True)
    parser.add_argument('--report_prefix', type=str, default='Contrastive Prompt Sweep')
    parser.add_argument('--temperature', type=float, default=0.6)
    parser.add_argument('--max_tokens', type=int, default=32768)
    parser.add_argument('--max_solution_chars', type=int, default=6000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max_model_len', type=int, default=8192)
    parser.add_argument('--max_num_batched_tokens', type=int, default=4096)
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.9)
    parser.add_argument('--abstraction_lora_path', type=str, default='')
    parser.add_argument('--solver_lora_path', type=str, default='')
    parser.add_argument('--judge_lora_path', type=str, default='')
    parser.add_argument('--compute_adherence', action='store_true')
    parser.add_argument('--label_enable_thinking', action='store_true')
    parser.add_argument('--skip_solver', action='store_true')
    return parser.parse_args()


def repo_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def run_command(cmd: list[str]) -> None:
    print('Running:', ' '.join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    output_root = repo_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    results = []

    for prompt_path_str in args.prompt_paths:
        prompt_path = repo_path(prompt_path_str)
        run_name = prompt_path.stem
        run_dir = output_root / run_name
        eval_cmd = [
            sys.executable,
            str(REPO_ROOT / 'scripts' / 'eval_contrastive_abstractions.py'),
            '--dataset_path', args.dataset_path,
            '--split', args.split,
            '--max_rows', str(args.max_rows),
            '--seed', str(args.seed),
            '--base_model', args.base_model,
            '--output_dir', str(run_dir),
            '--report_title', f'{args.report_prefix}: {run_name}',
            '--temperature', str(args.temperature),
            '--max_tokens', str(args.max_tokens),
            '--max_solution_chars', str(args.max_solution_chars),
            '--label_mode', args.label_mode,
            '--abstraction_prompt_path', str(prompt_path),
            '--few_shots_path', args.few_shots_path,
            '--solver_prompt_path', args.solver_prompt_path,
            '--max_model_len', str(args.max_model_len),
            '--max_num_batched_tokens', str(args.max_num_batched_tokens),
            '--gpu_memory_utilization', str(args.gpu_memory_utilization),
        ]
        if args.abstraction_lora_path:
            eval_cmd.extend(['--abstraction_lora_path', args.abstraction_lora_path])
        if args.solver_lora_path:
            eval_cmd.extend(['--solver_lora_path', args.solver_lora_path])
        if args.judge_lora_path:
            eval_cmd.extend(['--judge_lora_path', args.judge_lora_path])
        if args.compute_adherence:
            eval_cmd.append('--compute_adherence')
        if args.label_enable_thinking:
            eval_cmd.append('--label_enable_thinking')
        if args.skip_solver:
            eval_cmd.append('--skip_solver')
        run_command(eval_cmd)
        run_command([sys.executable, str(REPO_ROOT / 'scripts' / 'rescore_contrastive_eval.py'), '--run_dir', str(run_dir)])

        raw_report = json.loads((run_dir / 'report.json').read_text())
        rescored_report = json.loads((run_dir / 'rescored_report.json').read_text())
        result = {
            'prompt_path': str(prompt_path),
            'run_dir': str(run_dir),
            'raw_report': raw_report,
            'rescored_report': rescored_report,
        }
        results.append(result)

    results.sort(key=lambda item: item['rescored_report']['accuracy'], reverse=True)
    summary = {'results': results}
    (output_root / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')

    lines = ['# Contrastive Prompt Sweep', '']
    for item in results:
        report = item['rescored_report']
        lines.extend([
            f"## {Path(item['prompt_path']).name}",
            f"- accuracy: {report['num_correct']}/{report['num_rows']} = {report['accuracy']:.4f}",
            f"- baseline passrate mean: {report['baseline_passrate_mean']:.4f}",
            f"- absolute lift: {report['absolute_accuracy_lift']:.4f}",
            f"- run dir: `{item['run_dir']}`",
            '',
        ])
    (output_root / 'summary.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
