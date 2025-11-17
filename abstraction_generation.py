import os
from openai import OpenAI
import json
import time
import re
from typing import List, Dict
from tqdm import tqdm

def find_subsequence(text: str, search_sequence: str, wildcard: str = '...') -> List[Dict]:
    """
    Find subsequences in `text` that match `search_sequence` where occurrences
    of `wildcard` in `search_sequence` stand for any amount of text.
    Returns list of dicts: {'match': matched_substring, 'start': int, 'end': int}.
    """
    parts = search_sequence.split(wildcard)
    # escape literal parts so punctuation/regex chars are matched literally
    escaped_parts = [re.escape(p) for p in parts]
    # join with non-greedy wildcard; use re.DOTALL so '.' matches newlines
    pattern = '.*?'.join(escaped_parts)
    regex = re.compile(pattern, re.DOTALL)
    results = []
    for m in regex.finditer(text):
        results.append({'match': m.group(0), 'start': m.start(), 'end': m.end()})
    return results


openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

with open("abstraction_system_prompt.txt", 'r') as fp:
    lines = fp.readlines()

system_prompt = "".join(lines)

with open("abstraction_user_prompt.txt", 'r') as fp:
    lines = fp.readlines()

user_prompt = "".join(lines)

MODEL_NAME = "Qwen/Qwen3-1.7B"
qwen_17_path = "/scratch/rst306/verifier_scaling/verifiers/result/aime/Qwen_Qwen3-1.7B/20251002_134510/record.json"
qwen_8b_path = "/scratch/rst306/verifier_scaling/verifiers/result/aime/Qwen_Qwen3-8B/20251010_115527/record.json"
record_path = qwen_8b_path
with open(record_path, 'r') as fp:
    records = json.load(fp)
correct = [r for r in records if r['solver_correct']]
all_correct_outputs = [r['solver_full_output'] for r in correct]
# trace = all_correct_outputs[0]

# print("USER PROMPT = ", user_prompt.replace("REASONING_TRACES", trace))
# response = client.responses.create(
#     model="gpt-5-mini",
#     instructions=system_prompt,
#     input=user_prompt.replace("REASONING_TRACES", trace),
# )
# abstractions = json.loads(response.output_text)["abstractions"]
# print(response.output_text)
# last_abstraction = abstractions[-1]

# # search_string = last_abstraction["evidence_span_list"][-1]["text"]
# # subsequences = find_subsequence(trace, search_string)
# abs_name = last_abstraction["name"]
# abs_trigger = last_abstraction["trigger"]
# abs_procedure = "\n".join(last_abstraction["procedure"])
# abs_boundary_str = f"Abstraction Name: {abs_name}\n\nAbstraction Trigger: {abs_trigger}\n\nAbstraction Deliverables: {abs_procedure}\n\n"
# print(abs_boundary_str)

with open("boundary_prediction_system_prompt.txt", 'r') as fp:
    lines = fp.readlines()

boundary_system_prompt = "".join(lines)

with open("boundary_prediction_user_prompt.txt", 'r') as fp:
    lines = fp.readlines()

boundary_user_prompt = "".join(lines)


# response = client.responses.create(
#     model="gpt-5-mini",
#     instructions=boundary_system_prompt,
#     input=boundary_user_prompt.replace("{{ABSTRACTION}}", abs_boundary_str).replace("{{RAW_TEXT}}", trace),
# )

# print(response.output_text)
# boundary_output = json.loads(response.output_text)
# print("boundaries ", boundary_output["boundaries"])
# search_string = boundary_output["boundaries"][0]["boundary"].replace("«", "").replace("»", "")
# subsequences = find_subsequence(trace, search_string)
# print("search ", search_string)
# print("found ", subsequences)
output = []

for corr in tqdm(all_correct_outputs[:2]):
    trace = corr
    response = client.responses.create(
    model="gpt-5-mini",
    instructions=system_prompt,
    input=user_prompt.replace("REASONING_TRACES", trace),
    )

    abstractions = json.loads(response.output_text)["abstractions"]
    for i, abstraction in enumerate(abstractions):
        abs_name = abstraction["name"]
        abs_trigger = abstraction["trigger"]
        abs_procedure = "\n".join(abstraction["procedure"])
        abs_boundary_str = f"Abstraction Name: {abs_name}\n\nAbstraction Trigger: {abs_trigger}\n\nAbstraction Deliverables: {abs_procedure}\n\n"
        print(abs_boundary_str)

        boundary_user_prompt = "".join(lines)


        response = client.responses.create(
            model="gpt-5-mini",
            instructions=boundary_system_prompt,
            input=boundary_user_prompt.replace("{{ABSTRACTION}}", abs_boundary_str).replace("{{RAW_TEXT}}", trace),
        )

        boundary_output = json.loads(response.output_text)
        all_subsequence_matches = []
        for bound in boundary_output["boundaries"]:
            search_string = bound["boundary"].replace("«", "").replace("»", "")
            subsequences = find_subsequence(trace, search_string)
            all_subsequence_matches += subsequences

        abstractions[i]["boundaries"] = boundary_output["boundaries"]
        abstractions[i]["matched_subsequences"] = all_subsequence_matches
    
    out_dict = {"solver_output": trace, "abstractions": abstractions}
    output.append(out_dict)

save_dir = f"abstraction_results/{MODEL_NAME}"
os.makedirs(save_dir, exist_ok=True)
with open(f"{save_dir}/abstractions.json", 'w') as fp:
    json.dump(output, fp)
