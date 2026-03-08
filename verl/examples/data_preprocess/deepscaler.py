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
    parser.add_argument("--overfitting", action="store_true")

    args = parser.parse_args()
    local_dataset_path = args.local_dataset_path

    data_source = "deepscaler"
    
    with open("/scratch/rst306/action_abstractions/action_abstraction/prompt_templates/hint_conditioned_problem_solving.txt", "r") as fp:
        sft_prompt_template = fp.read()

    if local_dataset_path is not None:
        dataset = datasets.load_from_disk(local_dataset_path)
    else:
        dataset = datasets.load_dataset(data_source)
    
    dataset = dataset.filter(lambda ex: ex["abstraction"] is not None)
    dataset = dataset.filter(lambda ex: ex["abstraction"].strip() != "..." and ex["abstraction"].strip() != "")
    if args.overfitting:
        pts = ['In a new diagram, $A$ is the center of a circle with radii $AB=AC=8$. The sector $BOC$ is shaded except for a triangle $ABC$ within it, where $B$ and $C$ lie on the circle. If the central angle of $BOC$ is $240^\\circ$, what is the perimeter of the shaded region?',
            'In a new diagram, $A$ is the center of a circle with radii $AB=AC=8$. The sector $BOC$ is shaded except for a triangle $ABC$ within it, where $B$ and $C$ lie on the circle. If the central angle of $BOC$ is $240^\\circ$, what is the perimeter of the shaded region?',
            'In a new diagram, $A$ is the center of a circle with radii $AB=AC=8$. The sector $BOC$ is shaded except for a triangle $ABC$ within it, where $B$ and $C$ lie on the circle. If the central angle of $BOC$ is $240^\\circ$, what is the perimeter of the shaded region?',
            'In a new diagram, $A$ is the center of a circle with radii $AB=AC=8$. The sector $BOC$ is shaded except for a triangle $ABC$ within it, where $B$ and $C$ lie on the circle. If the central angle of $BOC$ is $240^\\circ$, what is the perimeter of the shaded region?',
            'In a new diagram, $A$ is the center of a circle with radii $AB=AC=8$. The sector $BOC$ is shaded except for a triangle $ABC$ within it, where $B$ and $C$ lie on the circle. If the central angle of $BOC$ is $240^\\circ$, what is the perimeter of the shaded region?',
            'In a new diagram, $A$ is the center of a circle with radii $AB=AC=8$. The sector $BOC$ is shaded except for a triangle $ABC$ within it, where $B$ and $C$ lie on the circle. If the central angle of $BOC$ is $240^\\circ$, what is the perimeter of the shaded region?',
            'There are 456 natives on an island, each of whom is either a knight who always tells the truth or a liar who always lies. All residents have different heights. Once, each native said, "All other residents are shorter than me!" What is the maximum number of natives who could have then said one minute later, "All other residents are taller than me?"',
            'There are 456 natives on an island, each of whom is either a knight who always tells the truth or a liar who always lies. All residents have different heights. Once, each native said, "All other residents are shorter than me!" What is the maximum number of natives who could have then said one minute later, "All other residents are taller than me?"',
            'There are 456 natives on an island, each of whom is either a knight who always tells the truth or a liar who always lies. All residents have different heights. Once, each native said, "All other residents are shorter than me!" What is the maximum number of natives who could have then said one minute later, "All other residents are taller than me?"',
            'There are 456 natives on an island, each of whom is either a knight who always tells the truth or a liar who always lies. All residents have different heights. Once, each native said, "All other residents are shorter than me!" What is the maximum number of natives who could have then said one minute later, "All other residents are taller than me?"',
            'There are 456 natives on an island, each of whom is either a knight who always tells the truth or a liar who always lies. All residents have different heights. Once, each native said, "All other residents are shorter than me!" What is the maximum number of natives who could have then said one minute later, "All other residents are taller than me?"',
            'There are 456 natives on an island, each of whom is either a knight who always tells the truth or a liar who always lies. All residents have different heights. Once, each native said, "All other residents are shorter than me!" What is the maximum number of natives who could have then said one minute later, "All other residents are taller than me?"']

        dataset["train"] = dataset["train"].filter(lambda ex: ex["problem"] in set(pts))
        dataset["test"] = dataset["train"]

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            problem = example["problem"]
            abstraction = example["abstraction"]

            prompt = sft_prompt_template.replace("{{PROBLEM}}", problem).replace("{{ABSTRACTION}}", abstraction.strip())

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
                    "split": split,
                    "index": idx,
                    "answer": solution,
                    "question": problem,
                },
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    hdfs_dir = args.hdfs_dir
    local_save_dir = args.local_dir
    if local_save_dir is not None:
        print("Warning: Argument 'local_dir' is deprecated. Please use 'local_save_dir' instead.")
    else:
        local_save_dir = args.local_save_dir
        if args.overfitting:
            local_save_dir = local_save_dir + "_overfitting"

    train_dataset.to_parquet(os.path.join(local_save_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_save_dir, "test.parquet"))

