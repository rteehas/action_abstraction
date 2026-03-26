from __future__ import annotations
import sys
from pathlib import Path

import torch
from datasets import load_from_disk
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

REPO_ROOT = Path('/workspace/action_abstraction')
SCRIPTS_DIR = REPO_ROOT / 'scripts'
for p in (REPO_ROOT, SCRIPTS_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from contrastive_abstraction_utils import read_text, strip_think_blocks

MODEL_NAME = 'Qwen/Qwen3-1.7B'
ds = load_from_disk(str(REPO_ROOT / 'contrastive_abstraction_datasets/deepscaler_mixed_correct_incorrect_256_all_correct'))['test']
row = dict(ds[0])
trace = strip_think_blocks(row['selected_correct_solutions'][0])
base = read_text(REPO_ROOT / 'prompt_templates/principle_extraction_template.txt').replace('{{PROBLEM}}', row['problem']).replace('{{CORRECT_SOLUTIONS_BLOCK}}', trace)
variants = {
    'base': base,
    'principles_suffix': base + '\n\nPRINCIPLES:\n',
    'assistant_suffix': base + '\n\nExtract 2 to 5 principles now.\n',
}

def main() -> None:
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    msgs = [tok.apply_chat_template([{'role':'user','content':p}], tokenize=False, add_generation_prompt=True, enable_thinking=False) for p in variants.values()]
    print('Device name =', torch.cuda.get_device_name(0))
    llm = LLM(model=MODEL_NAME, enable_lora=False, max_model_len=16384, max_num_batched_tokens=2048, gpu_memory_utilization=0.9)
    samp = SamplingParams(temperature=0.0, top_p=1.0, top_k=-1, max_tokens=512, seed=0)
    outs = llm.generate(msgs, samp)
    for (name, _), out in zip(variants.items(), outs):
        text = (out.outputs[0].text or '').strip()
        print('VARIANT', name)
        print(repr(text[:1200]))
        print('---')

if __name__ == '__main__':
    main()
