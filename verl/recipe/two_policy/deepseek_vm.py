from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, Qwen2Model, Qwen2PreTrainedModel
from transformers.cache_utils import Cache
from transformers.modeling_outputs import SequenceClassifierOutputWithPast


DEEPSEEK_ROLLIN_SUFFIX = " Please think step-by-step and put your final answer within \\boxed{}."


def format_roll_in(problem: str) -> str:
    return f"<｜begin▁of▁sentence｜><｜User｜>{problem}{DEEPSEEK_ROLLIN_SUFFIX}<｜Assistant｜><think>\n"


def parse_dtype(name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping[name]


@dataclass
class ClassifierOutputWithPast(SequenceClassifierOutputWithPast):
    success_probs: torch.FloatTensor | None = None


class Qwen2ForClassifier(Qwen2PreTrainedModel):
    """Inference-oriented classifier head used by DeepSeek-VM checkpoints."""

    def __init__(self, config, use_bias: bool = False):
        super().__init__(config)
        self.model = Qwen2Model(config)
        self.num_labels = int(config.num_labels)
        self.score = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size, bias=use_bias),
            nn.ReLU(),
            nn.Linear(config.hidden_size, self.num_labels, bias=use_bias),
        )
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value) -> None:
        self.model.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> ClassifierOutputWithPast:
        return_dict = True if return_dict is None else return_dict
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        logits = self.score(hidden_states).float()
        if logits.shape[-1] == 1:
            success_probs = logits.squeeze(-1).sigmoid()
        else:
            success_probs = F.softmax(logits, dim=-1)[..., 1]
        return ClassifierOutputWithPast(
            logits=logits,
            success_probs=success_probs,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class DeepSeekVMScorer:
    def __init__(
        self,
        *,
        model_path: str,
        tokenizer_path: str,
        torch_dtype: str,
        attn_implementation: str,
        device: str | None,
        device_map: str | None,
        batch_size: int,
        offload_enable: bool,
        offload_folder: str | None,
        offload_state_dict: bool,
        low_cpu_mem_usage: bool,
    ):
        self.batch_size = max(1, int(batch_size))
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, padding_side="left")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.model_max_length = max(int(getattr(self.tokenizer, "model_max_length", 0)), 131072)

        dtype = parse_dtype(torch_dtype)
        effective_device_map = device_map
        if offload_enable and (device_map is None or str(device_map).lower() == "none"):
            effective_device_map = "auto"
        model_kwargs = {
            "dtype": dtype,
            "use_cache": False,
            "low_cpu_mem_usage": bool(low_cpu_mem_usage),
        }
        if effective_device_map and str(effective_device_map).lower() != "none":
            model_kwargs["device_map"] = effective_device_map
        if offload_enable:
            model_kwargs["offload_state_dict"] = bool(offload_state_dict)
            if offload_folder:
                os.makedirs(offload_folder, exist_ok=True)
                model_kwargs["offload_folder"] = offload_folder
        # Flash attention only works on CUDA — fall back to eager on CPU
        target_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if target_device.type == "cpu" and attn_implementation in ("flash_attention_2",):
            attn_implementation = "eager"
        try:
            self.model = Qwen2ForClassifier.from_pretrained(
                model_path,
                attn_implementation=attn_implementation,
                **model_kwargs,
            )
        except Exception:
            self.model = Qwen2ForClassifier.from_pretrained(model_path, **model_kwargs)

        if not effective_device_map or str(effective_device_map).lower() == "none":
            target_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
            self.model = self.model.to(target_device)
            self.model_device = target_device
        else:
            self.model_device = self._resolve_model_input_device(explicit_device=device)
        self.model.eval()
        self.model.config.use_cache = False

    def _resolve_model_input_device(self, explicit_device: str | None) -> torch.device:
        if explicit_device:
            return torch.device(explicit_device)

        hf_device_map = getattr(self.model, "hf_device_map", None) or {}
        for mapped_device in hf_device_map.values():
            if mapped_device in {"cpu", "disk", "meta"}:
                continue
            if isinstance(mapped_device, int):
                return torch.device(f"cuda:{mapped_device}")
            return torch.device(str(mapped_device))

        return torch.device("cpu")

    @lru_cache(maxsize=16384)
    def _roll_in_ids(self, problem: str) -> tuple[int, ...]:
        return tuple(self.tokenizer.encode(format_roll_in(problem), add_special_tokens=False))

    def score_partial_texts(self, problems: list[str], responses: list[str]) -> tuple[list[float], list[int]]:
        if not problems:
            return [], []

        scores: list[float] = []
        observed_tokens: list[int] = []
        pad_token_id = int(self.tokenizer.pad_token_id)

        for start in range(0, len(problems), self.batch_size):
            batch_problems = problems[start : start + self.batch_size]
            batch_responses = responses[start : start + self.batch_size]

            input_ids_list: list[list[int]] = []
            batch_observed_tokens: list[int] = []
            for problem, response in zip(batch_problems, batch_responses, strict=True):
                response_ids = self.tokenizer.encode(response, add_special_tokens=False)
                batch_observed_tokens.append(len(response_ids))
                input_ids_list.append(list(self._roll_in_ids(problem)) + response_ids)

            max_len = max(len(item) for item in input_ids_list)
            input_ids = torch.full((len(input_ids_list), max_len), pad_token_id, dtype=torch.long)
            attention_mask = torch.zeros((len(input_ids_list), max_len), dtype=torch.long)
            for idx, token_ids in enumerate(input_ids_list):
                seq = torch.tensor(token_ids, dtype=torch.long)
                input_ids[idx, -len(token_ids) :] = seq
                attention_mask[idx, -len(token_ids) :] = 1

            enc = {
                "input_ids": input_ids.to(self.model_device),
                "attention_mask": attention_mask.to(self.model_device),
            }
            with torch.inference_mode():
                outputs = self.model(**enc, return_dict=True, use_cache=False)
                success_probs = outputs.success_probs
                last_indices = enc["attention_mask"].sum(dim=1) - 1
                batch_indices = torch.arange(success_probs.shape[0], device=success_probs.device)
                batch_scores = success_probs[batch_indices, last_indices].detach().float().cpu().tolist()

            scores.extend(float(score) for score in batch_scores)
            observed_tokens.extend(batch_observed_tokens)

        return scores, observed_tokens
