import argparse

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-model', default='Qwen/Qwen3-1.7B')
    parser.add_argument(
        '--lora-path',
        default='/workspace/action_abstraction/sft_models/Qwen3_1_7B-principle_generation_non_regressed_9k_subset_all_correct_v5_seed100_prompt_sft_principle_generation_half_epoch_eval/20260322_023906/checkpoint-1736',
    )
    parser.add_argument(
        '--output-dir',
        default='/workspace/action_abstraction/merged_models/qwen3_1_7b_principle_generator_ckpt1736',
    )
    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype='auto', device_map='cpu')
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    merged = PeftModel.from_pretrained(model, args.lora_path).merge_and_unload()
    merged.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == '__main__':
    main()
