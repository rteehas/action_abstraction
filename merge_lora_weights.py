from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datetime import datetime

BASE = "Qwen/Qwen3-1.7B"
SFT_LORA = "/scratch/rst306/action_abstractions/action_abstraction/sft_models/Qwen3_1_7B-solver/20260306_093750/checkpoint-450"
OUT = "solver_sft_qwen_1_7b_450"

base = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype="auto", device_map="cpu")
tok = AutoTokenizer.from_pretrained(BASE)

sft = PeftModel.from_pretrained(base, SFT_LORA)
merged = sft.merge_and_unload()
merged.save_pretrained(OUT, safe_serialization=True)
tok.save_pretrained(OUT)
