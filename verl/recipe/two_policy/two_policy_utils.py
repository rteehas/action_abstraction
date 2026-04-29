import re
from dataclasses import dataclass

import numpy as np
import torch


def load_template(path: str) -> str:
    with open(path) as f:
        return f.read()


def render_template(template: str, problem: str, abstraction: str | None = None) -> str:
    rendered = template.replace('{{PROBLEM}}', problem)
    if abstraction is not None:
        rendered = rendered.replace('{{ABSTRACTION}}', abstraction)
    return rendered


def build_single_turn_chat(content: str) -> list[dict]:
    return [{'role': 'user', 'content': content}]


def as_object_array(values) -> np.ndarray:
    return np.array(list(values), dtype=object)


@dataclass
class AbstractionParseResult:
    raw_text: str
    abstraction: str | None
    solver_conditioning_abstraction: str | None
    is_valid: bool
    validity_score: float
    failure_reason: str | None
    principle_count: int


def strip_abstraction_tags(text: str, open_tag: str, close_tag: str) -> str:
    return text.replace(open_tag, '\n').replace(close_tag, '\n').strip()


def normalize_solver_conditioning_abstraction(text: str | None, invalid_placeholders: tuple[str, ...]) -> str | None:
    if text is None:
        return None
    cleaned = text.strip()
    if not cleaned:
        return None
    if cleaned.lower() in invalid_placeholders:
        return None
    return cleaned


def parse_abstraction_output(
    text: str,
    open_tag: str,
    close_tag: str,
    min_chars: int,
    invalid_placeholders: tuple[str, ...],
    require_principle_headers: bool,
    max_num_principles: int | None,
) -> AbstractionParseResult:
    pattern = re.escape(open_tag) + r'([\s\S]*?)' + re.escape(close_tag)
    matches = re.findall(pattern, text)
    fallback_solver_abstraction = normalize_solver_conditioning_abstraction(
        strip_abstraction_tags(text, open_tag, close_tag),
        invalid_placeholders,
    )
    if len(matches) != 1:
        return AbstractionParseResult(
            raw_text=text,
            abstraction=None,
            solver_conditioning_abstraction=fallback_solver_abstraction,
            is_valid=False,
            validity_score=0.0,
            failure_reason='expected_exactly_one_tagged_abstraction',
            principle_count=0,
        )

    abstraction = matches[0].strip()
    solver_conditioning_abstraction = normalize_solver_conditioning_abstraction(abstraction, invalid_placeholders)
    if len(abstraction) < min_chars:
        return AbstractionParseResult(
            raw_text=text,
            abstraction=abstraction,
            solver_conditioning_abstraction=solver_conditioning_abstraction,
            is_valid=False,
            validity_score=0.0,
            failure_reason='abstraction_too_short',
            principle_count=0,
        )

    if abstraction.strip().lower() in invalid_placeholders:
        return AbstractionParseResult(
            raw_text=text,
            abstraction=abstraction,
            solver_conditioning_abstraction=None,
            is_valid=False,
            validity_score=0.0,
            failure_reason='placeholder_abstraction',
            principle_count=0,
        )

    principle_count = len(re.findall(r'(?m)^\[(?:CORE|SUPPORT)\]\s*$', abstraction))
    required_headers = ('Title:', 'What:', 'Background:', 'When:', 'Why:', 'Role:')
    has_headers = all(header in abstraction for header in required_headers)

    if require_principle_headers and (principle_count == 0 or not has_headers):
        return AbstractionParseResult(
            raw_text=text,
            abstraction=abstraction,
            solver_conditioning_abstraction=solver_conditioning_abstraction,
            is_valid=True,
            validity_score=0.5,
            failure_reason='missing_expected_principle_structure',
            principle_count=principle_count,
        )

    if max_num_principles is not None and principle_count > max_num_principles:
        return AbstractionParseResult(
            raw_text=text,
            abstraction=abstraction,
            solver_conditioning_abstraction=solver_conditioning_abstraction,
            is_valid=True,
            validity_score=0.5,
            failure_reason='too_many_principles',
            principle_count=principle_count,
        )

    return AbstractionParseResult(
        raw_text=text,
        abstraction=abstraction,
        solver_conditioning_abstraction=solver_conditioning_abstraction,
        is_valid=True,
        validity_score=1.0,
        failure_reason=None,
        principle_count=principle_count,
    )


def build_scalar_reward_tensor(response_mask: torch.Tensor, scalar_rewards: torch.Tensor) -> torch.Tensor:
    reward_tensor = torch.zeros_like(response_mask, dtype=torch.float32)
    valid_lengths = response_mask.sum(dim=1).to(torch.long)
    reward_tensor[torch.arange(reward_tensor.size(0)), valid_lengths - 1] = scalar_rewards.to(torch.float32)
    return reward_tensor


def aggregate_child_rewards(values: list[float], mode: str, mean_weight: float, max_weight: float) -> float:
    if not values:
        return 0.0

    if mode == 'mean':
        return float(np.mean(values))
    if mode == 'max':
        return float(np.max(values))
    if mode == 'weighted_mean_max':
        return float(mean_weight * np.mean(values) + max_weight * np.max(values))

    raise ValueError(f'Unsupported aggregation mode: {mode}')
