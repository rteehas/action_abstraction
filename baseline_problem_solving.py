from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from datasets import load_from_disk, load_dataset
import json
from tqdm import tqdm
from argparse import ArgumentParser
import random


def make_problem_solving_prompt(problem):
    with open("/scratch/rst306/action_abstractions/action_abstraction/prompt_templates/baseline_problem_solving.txt", 'r') as fp:
        lines = fp.readlines()
    
    problem_solving_prompt_template = "".join(lines)

    prompt = problem_solving_prompt_template.replace("{{PROBLEM}}", problem)
    return prompt

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_solutions", type=int, default=1)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    # parser.add_argument("--token_budget", type=int, default=8000)
    return parser

if __name__ == "__main__":

    args = get_args().parse_args()
    print("Args:")
    print(args)

    # dataset = load_dataset("TianHongZXY/aime-1983-2025", split='test')
    dataset = load_dataset("agentica-org/DeepScaleR-Preview-Dataset", split="train")
    shard_size = dataset.num_rows // args.num_shards
    n = args.shard
    start = n * shard_size
    end = dataset.num_rows if n == args.num_shards - 1 else (n + 1) * shard_size  # last shard gets remainder

    shard_n = dataset.select(range(start, end))

    model_name = "Qwen/Qwen3-1.7B"
    # 3. Sample indices from the original dataset
    random.seed(args.seed)  # for reproducibility
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    texts = []
    

    for example in tqdm(shard_n, desc="creating messages"):

        problem = example['problem']
        prompt = make_problem_solving_prompt(problem)
        # Initialize the tokenizer
        
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,  # Set to False to strictly disable thinking
        )
        texts.append(text)

    

    # Configurae the sampling parameters (for thinking mode)
    sampling_params = SamplingParams(temperature=0.6, top_p=0.95, top_k=20, max_tokens=8000, n=args.n_solutions)

    # Initialize the vLLM engine
    llm = LLM(model=model_name, dtype="bfloat16",tensor_parallel_size=1)

    # Generate outputs
    outputs = llm.generate(texts, sampling_params)

    problem_solutions = []
    for output in outputs:
        solns = []
        for out_p in output.outputs:
            solns.append(out_p.text)
        problem_solutions.append(solns)

    # Sanity check: lengths must match
    assert len(problem_solutions) == len(shard_n), \
        f"Length mismatch: {len(problem_solutions)} generations vs {len(shard_n)} examples"

    # Add as new column and save dataset
    ds_with_generations = shard_n.add_column(
        "generated_solution", problem_solutions
    )

    # ds_with_generations.save_to_disk(f"deepscaler_baseline_solutions_shard_{args.shard}_of_{args.num_shards - 1}")
    ds_with_generations.save_to_disk(f"aime_qwen_1_7_baseline_solutions_shard_{args.shard}_of_{args.num_shards - 1}")
