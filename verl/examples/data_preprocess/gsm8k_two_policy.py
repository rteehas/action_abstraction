# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the GSM8K dataset to parquet format for two-policy training.

This variant follows the baseline GSM8K preprocessing flow:
- load raw GSM8K from Hugging Face or an on-disk DatasetDict
- optionally carve out a validation split from the train split
- preserve the original GSM8K test split

It writes a standard VERL-compatible ``prompt`` column for abstraction-stage
schema compatibility and also stores explicit prompt payloads for both the
abstraction generator and solver template.
"""

import argparse
import os
import re
from pathlib import Path

import datasets

from verl.utils.hdfs_io import copy, makedirs


DEFAULT_RAW_DATA_SOURCE = "openai/gsm8k"
DEFAULT_REWARD_DATA_SOURCE = "math"
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ABSTRACTION_TEMPLATE_PATH = (
    REPO_ROOT / "prompt_templates" / "insight_abstraction_generation_sft_template.txt"
)
DEFAULT_SOLVER_TEMPLATE_PATH = REPO_ROOT / "prompt_templates" / "hint_conditioned_problem_solving_rich_v1.txt"


def extract_solution(solution_str: str) -> str:
    solution = re.search(r"#### (\-?[0-9\.\,]+)", solution_str)
    assert solution is not None
    final_solution = solution.group(0)
    final_solution = final_solution.split("#### ")[1].replace(",", "")
    return final_solution


def parse_validation_size(value):
    if value is None:
        return None

    try:
        parsed_value = int(value)
    except ValueError:
        try:
            parsed_value = float(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                "validation size must be either a positive integer count or a float in (0, 1)"
            ) from exc
        if not 0 < parsed_value < 1:
            raise argparse.ArgumentTypeError("float validation size must be in the open interval (0, 1)")
        return parsed_value

    if parsed_value <= 0:
        raise argparse.ArgumentTypeError("integer validation size must be positive")

    return parsed_value


def parse_passrate_threshold(value):
    if value is None:
        return None

    try:
        parsed_value = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("passrate threshold must be a float in [0, 1]") from exc

    if not 0 <= parsed_value <= 1:
        raise argparse.ArgumentTypeError("passrate threshold must lie in [0, 1]")

    return parsed_value


def load_dataset(local_dataset_path, data_source):
    if local_dataset_path is None:
        return datasets.load_dataset(data_source, "main")

    expanded_path = Path(os.path.expanduser(local_dataset_path))
    if expanded_path.exists() and (expanded_path / "dataset_dict.json").exists():
        return datasets.load_from_disk(str(expanded_path))

    return datasets.load_dataset(local_dataset_path, "main")


def render_template(template: str, problem: str, abstraction: str | None = None) -> str:
    rendered = template.replace("{{PROBLEM}}", problem)
    rendered = rendered.replace("{{QUESTION}}", problem)
    if abstraction is not None:
        rendered = rendered.replace("{{ABSTRACTION}}", abstraction)
    return rendered


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default=None, help="Deprecated alias for --local_save_dir.")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--local_dataset_path", default=None, help="The local path to the raw dataset, if it exists.")
    parser.add_argument(
        "--local_save_dir",
        default="~/data/gsm8k_two_policy",
        help="The save directory for the preprocessed dataset.",
    )
    parser.add_argument(
        "--abstraction_prompt_template_path",
        default=str(DEFAULT_ABSTRACTION_TEMPLATE_PATH),
        help="Path to the abstraction prompt template containing the {{PROBLEM}} placeholder.",
    )
    parser.add_argument(
        "--solver_prompt_template_path",
        default=str(DEFAULT_SOLVER_TEMPLATE_PATH),
        help=(
            "Path to the solver prompt template containing the {{PROBLEM}} placeholder. "
            "If the template also contains {{ABSTRACTION}}, that placeholder is preserved in the saved column."
        ),
    )
    parser.add_argument(
        "--reward_data_source",
        default=DEFAULT_REWARD_DATA_SOURCE,
        help=(
            "Data source tag used by the reward scorer. Default is 'math' because the "
            "two-policy solver template expects answers in \\boxed{} format."
        ),
    )
    parser.add_argument(
        "--validation_size",
        type=parse_validation_size,
        default=None,
        help=(
            "Optional held-out split carved from GSM8K train. Accepts either a positive integer count or a float "
            "fraction in (0, 1)."
        ),
    )
    parser.add_argument(
        "--validation_seed",
        type=int,
        default=42,
        help="Random seed used when creating the held-out validation split from train.",
    )
    parser.add_argument(
        "--max_train_passrate",
        type=parse_passrate_threshold,
        default=None,
        help=(
            "If set, keep only train examples whose source passrate is less than or equal to this threshold "
            "before any validation split is carved out. Requires a 'passrate' column in the source train split."
        ),
    )

    args = parser.parse_args()

    raw_dataset = load_dataset(args.local_dataset_path, DEFAULT_RAW_DATA_SOURCE)
    if not isinstance(raw_dataset, datasets.DatasetDict):
        raise ValueError("Expected a DatasetDict with a 'train' split and, optionally, a 'test' split.")
    if "train" not in raw_dataset:
        raise ValueError("Expected the dataset to contain a 'train' split.")

    abstraction_prompt_template = Path(args.abstraction_prompt_template_path).read_text()
    solver_prompt_template = Path(args.solver_prompt_template_path).read_text()

    train_dataset = raw_dataset["train"]
    if args.max_train_passrate is not None:
        if "passrate" not in train_dataset.column_names:
            raise ValueError(
                "max_train_passrate was set, but the source train split does not contain a 'passrate' column."
            )
        train_dataset = train_dataset.filter(lambda example: example["passrate"] <= args.max_train_passrate)

    validation_dataset = None
    if args.validation_size is not None:
        split_dataset = train_dataset.train_test_split(test_size=args.validation_size, seed=args.validation_seed)
        train_dataset = split_dataset["train"]
        validation_dataset = split_dataset["test"]

    test_dataset = raw_dataset["test"] if "test" in raw_dataset else None

    def make_map_fn(split):
        def process_fn(example, idx):
            question_raw = example["question"]
            answer_raw = example["answer"]
            final_answer = extract_solution(answer_raw)

            abstraction_prompt = render_template(abstraction_prompt_template, question_raw)
            solver_prompt_template_text = render_template(solver_prompt_template, question_raw)

            extra_info = {
                "split": split,
                "index": idx,
                "answer": answer_raw,
                "question": question_raw,
                "raw_data_source": DEFAULT_RAW_DATA_SOURCE,
                "abstraction_prompt_template_path": args.abstraction_prompt_template_path,
                "solver_prompt_template_path": args.solver_prompt_template_path,
            }
            if "source_row_idx" in example:
                extra_info["source_row_idx"] = example["source_row_idx"]
            if "passrate" in example:
                extra_info["source_passrate"] = example["passrate"]
            if "num_correct" in example:
                extra_info["source_num_correct"] = example["num_correct"]

            return {
                "problem": question_raw,
                "answer": final_answer,
                "prompt": [
                    {
                        "role": "user",
                        "content": abstraction_prompt,
                    }
                ],
                "abstraction_prompt": abstraction_prompt,
                "solver_prompt_template": solver_prompt_template_text,
                "data_source": args.reward_data_source,
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": final_answer},
                "extra_info": extra_info,
            }

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    if validation_dataset is not None:
        validation_dataset = validation_dataset.map(function=make_map_fn("validation"), with_indices=True)
    if test_dataset is not None:
        test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    hdfs_dir = args.hdfs_dir
    local_save_dir = args.local_dir
    if local_save_dir is not None:
        print("Warning: Argument 'local_dir' is deprecated. Please use 'local_save_dir' instead.")
    else:
        local_save_dir = args.local_save_dir

    local_save_dir = os.path.expanduser(local_save_dir)
    os.makedirs(local_save_dir, exist_ok=True)

    train_dataset.to_parquet(os.path.join(local_save_dir, "train.parquet"))
    if validation_dataset is not None:
        validation_dataset.to_parquet(os.path.join(local_save_dir, "val.parquet"))
    if test_dataset is not None:
        test_dataset.to_parquet(os.path.join(local_save_dir, "test.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_save_dir, dst=hdfs_dir)
