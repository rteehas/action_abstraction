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
Preprocess the GSM8k dataset to parquet format
"""

import argparse
import os
import re

import datasets

from verl.utils.hdfs_io import copy, makedirs


def extract_solution(solution_str):
    solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
    assert solution is not None
    final_solution = solution.group(0)
    final_solution = final_solution.split("#### ")[1].replace(",", "")
    return final_solution


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default=None, help="The save directory for the preprocessed dataset.")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--local_dataset_path", default=None, help="The local path to the raw dataset, if it exists.")
    parser.add_argument(
        "--local_save_dir", default="/data/deepscaler", help="The save directory for the preprocessed dataset."
    )

    args = parser.parse_args()
    local_dataset_path = args.local_dataset_path

    
    with open("/scratch/rst306/action_abstractions/action_abstraction/prompt_templates/hint_conditioned_problem_solving.txt", "r") as fp:
        sft_prompt_template = fp.read()

    if local_dataset_path is not None:
        dataset = datasets.load_from_disk(local_dataset_path)
    else:
        dataset = datasets.load_dataset(data_source)
    
    dataset = dataset.filter(lambda ex: ex["abstraction"] is not None)
    dataset = dataset.filter(lambda ex: ex["abstraction"].strip() != "..." and ex["abstraction"].strip() != "")

    # add a row to each data item that represents a unique id
    def process_fn(example, idx):
        problem = example["problem"]
        abstraction = example["abstraction"]

        prompt = sft_prompt_template.replace("{{PROBLEM}}", problem).replace("{{ABSTRACTION}}", abstraction.strip())
        data_source = example["source"]
        solution = example["answer"]            
        data = {
            "data_source": data_source,
            "prompt": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": solution},
            "extra_info": {
                "split": "test",
                "index": idx,
                "answer": solution,
                "question": problem,
            },
        }
        return data


    dataset = dataset.map(function=process_fn, with_indices=True)

    hdfs_dir = args.hdfs_dir
    local_save_dir = args.local_dir
    if local_save_dir is not None:
        print("Warning: Argument 'local_dir' is deprecated. Please use 'local_save_dir' instead.")
    else:
        local_save_dir = args.local_save_dir

    print("Train Example")
    print(dataset[0])
    dataset.to_parquet(os.path.join(local_save_dir, "test.parquet"))

