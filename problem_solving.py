from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from datasets import load_from_disk, load_dataset
import json
from tqdm import tqdm
from argparse import ArgumentParser
import random


def make_problem_solving_prompt(problem, abstraction):
    with open("/scratch/rst306/action_abstractions/action_abstraction/prompt_templates/abstraction_conditioned_problem_solving.txt", 'r') as fp:
        lines = fp.readlines()
    
    problem_solving_prompt_template = "".join(lines)

    prompt = problem_solving_prompt_template.replace("{{PROBLEM}}", problem).replace("{{ABSTRACTION}}", abstraction)
    return prompt

def make_problem_solving_prompt_with_procedure(problem, abstraction, procedure):
    with open("/scratch/rst306/action_abstractions/action_abstraction/prompt_templates/abstraction_procedure_conditioned_problem_solving.txt", 'r') as fp:
        lines = fp.readlines()

    problem_solving_prompt_template = "".join(lines)

    prompt = problem_solving_prompt_template.replace("{{PROBLEM}}", problem).replace("{{ABSTRACTION}}", abstraction).replace("{{PROCEDURE}}", procedure)

    return prompt

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_solutions", type=int, default=1)
    parser.add_argument("--with_procedure", action="store_true")
    parser.add_argument("--abstraction_column", type=str, default="abstraction")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    return parser

if __name__ == "__main__":

    args = get_args().parse_args()
    print("Args:")
    print(args)
    abstraction_column = args.abstraction_column
    # subset_dataset = load_from_disk("openmathreasoning_500_subset_with_abstractions_extracted")
    # to_remove = [92, 136, 459]
    if args.with_procedure:
        raise Exception("Reimplement")
        subset_dataset = load_from_disk("aime_abstractions_procedures_extracted")
    else:
        subset_dataset = load_from_disk("openmathreasoning_10k_subset_with_abstractions_extracted")

    ds_filtered = subset_dataset.filter(lambda ex: ex[abstraction_column] is not None and ex[abstraction_column].strip() != "" and ex[abstraction_column].strip() != "...")

    if args.with_procedure:
        ds_filtered = ds_filtered.filter(lambda ex: ex["procedure"] is not None)

    shard_size = ds_filtered.num_rows // args.num_shards
    n = args.shard
    start = n * shard_size
    end = ds_filtered.num_rows if n == args.num_shards - 1 else (n + 1) * shard_size  # last shard gets remainder

    shard_n = ds_filtered.select(range(start, end))

    base_model = "Qwen/Qwen3-1.7B"
    # 3. Sample indices from the original dataset
    random.seed(args.seed)  # for reproducibility
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    texts = []
    

    for example in tqdm(shard_n, desc="creating messages"):

        problem = example['problem']
        abstraction = example[abstraction_column] #example["abstraction"]
        if args.with_procedure:
            procedure = example["procedure"]
            prompt = make_problem_solving_prompt_with_procedure(problem, abstraction, procedure)

        else:
            prompt = make_problem_solving_prompt(problem, abstraction)
        # Initialize the tokenizer
        
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,  # Set to False to strictly disable thinking
        )
        texts.append(text)

    print(texts[0])
    

    # Configurae the sampling parameters (for thinking mode)
    sampling_params = SamplingParams(temperature=0.6, top_p=0.95, top_k=20, max_tokens=24000, n=args.n_solutions)

    # Initialize the vLLM engine
    llm = LLM(model=base_model)

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

    if args.with_procedure:
        # Add as new column and save dataset
        ds_with_generations = shard_n.add_column(
            "abstraction_procedure_conditioned_problem_solutions", problem_solutions
        )

        ds_with_generations.save_to_disk(f"aime_abstraction_procedure_conditioned_solutions")
    else:
        # Add as new column and save dataset
        ds_with_generations = shard_n.add_column(
            "abstraction_conditioned_problem_solutions", problem_solutions
        )

        ds_with_generations.save_to_disk(f"omr_10k_absconditioned_shard_{args.shard}_of_{args.num_shards - 1}")
