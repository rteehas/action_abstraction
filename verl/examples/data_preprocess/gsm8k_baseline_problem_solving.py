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
Preprocess the GSM8K dataset to parquet format using the baseline problem-solving template.

This variant can also carve out a held-out validation split from the training split while
preserving the original GSM8K test split for final evaluation.
"""

import argparse
import os
import re
from pathlib import Path

import datasets

from verl.utils.hdfs_io import copy, makedirs

DEFAULT_RAW_DATA_SOURCE = "openai/gsm8k"
DEFAULT_REWARD_DATA_SOURCE = "math"
DEFAULT_TEMPLATE_PATH = Path(__file__).resolve().parents[3] / "prompt_templates" / "baseline_problem_solving.txt"


def extract_solution(solution_str):
    solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
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


def load_dataset(local_dataset_path, data_source):
    if local_dataset_path is None:
        return datasets.load_dataset(data_source, "main")

    expanded_path = Path(os.path.expanduser(local_dataset_path))
    if expanded_path.exists() and (expanded_path / "dataset_dict.json").exists():
        return datasets.load_from_disk(str(expanded_path))

    return datasets.load_dataset(local_dataset_path, "main")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default=None, help="Deprecated alias for --local_save_dir.")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--local_dataset_path", default=None, help="The local path to the raw dataset, if it exists.")
    parser.add_argument(
        "--local_save_dir",
        default="~/data/gsm8k_baseline_problem_solving",
        help="The save directory for the preprocessed dataset.",
    )
    parser.add_argument(
        "--prompt_template_path",
        default=str(DEFAULT_TEMPLATE_PATH),
        help="Path to the prompt template containing the {{PROBLEM}} placeholder.",
    )
    parser.add_argument(
        "--reward_data_source",
        default=DEFAULT_REWARD_DATA_SOURCE,
        help=(
            "Data source tag used by the reward scorer. Default is 'math' because the baseline template expects "
            "answers in \\\\boxed{} format. Use 'openai/gsm8k' only if your prompt also expects #### answers."
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

    args = parser.parse_args()

    raw_dataset = load_dataset(args.local_dataset_path, DEFAULT_RAW_DATA_SOURCE)
    if not isinstance(raw_dataset, datasets.DatasetDict):
        raise ValueError("Expected a DatasetDict with a 'train' split and, optionally, a 'test' split.")
    if "train" not in raw_dataset:
        raise ValueError("Expected the dataset to contain a 'train' split.")

    prompt_template = Path(args.prompt_template_path).read_text()

    train_dataset = raw_dataset["train"]
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
            solution = extract_solution(answer_raw)

            prompt = prompt_template.replace("{{PROBLEM}}", question_raw)

            return {
                "data_source": args.reward_data_source,
                "prompt": [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": solution},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": answer_raw,
                    "question": question_raw,
                    "raw_data_source": DEFAULT_RAW_DATA_SOURCE,
                },
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
