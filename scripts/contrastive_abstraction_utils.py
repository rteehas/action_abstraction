from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List, Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]


def repo_path(*parts: str) -> Path:
    return REPO_ROOT.joinpath(*parts)


def read_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def extract_tagged(text: str, tag: str) -> Optional[str]:
    match = re.search(rf"<{tag}>([\s\S]*?)</{tag}>", text)
    if not match:
        return None
    return match.group(1).strip()


def extract_abstraction(text: str) -> Optional[str]:
    return extract_tagged(text, "abstraction")


def extract_last_boxed(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    matches = list(re.finditer(r"\\boxed\s*\{", text))
    if not matches:
        return None

    start = matches[-1].start()
    idx = matches[-1].end()
    depth = 1
    while idx < len(text) and depth > 0:
        char = text[idx]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
        idx += 1
    if depth != 0:
        return None
    return text[start:idx].strip()


def _normalize_answer_text(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    normalized = text
    normalized = normalized.replace('\\dfrac', '\\frac').replace('\\tfrac', '\\frac')
    normalized = normalized.replace('\\left', '').replace('\\right', '')
    return re.sub(r"\s+", "", normalized)


try:
    from math_verify import parse, verify  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    parse = None
    verify = None


def answers_match(expected: Optional[str], predicted: Optional[str]) -> bool:
    expected_norm = _normalize_answer_text(expected)
    predicted_norm = _normalize_answer_text(predicted)
    if not expected_norm or not predicted_norm:
        return False
    if expected_norm == predicted_norm:
        return True
    if parse is None or verify is None:
        return False
    try:
        return bool(verify(parse(expected_norm), parse(predicted_norm)))
    except Exception:
        return False


def classify_generated_answers(answer: str, generated_answers: Sequence[Optional[str]]) -> List[bool]:
    target = answer if "\\boxed" in answer else f"\\boxed{{{answer}}}"
    return [answers_match(target, candidate) for candidate in generated_answers]


def strip_think_blocks(text: str) -> str:
    has_open = "<think>" in text
    has_close = "</think>" in text
    if has_open != has_close:
        return ""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def shorten_text(text: str, max_chars: int = 6000) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 15].rstrip() + "\n...[truncated]"


def format_solution_block(label: str, solutions: Iterable[str], max_chars: int = 6000) -> str:
    rendered = []
    kept_idx = 1
    for solution in solutions:
        cleaned = strip_think_blocks(solution)
        if not cleaned:
            continue
        rendered.append(f"[{label} {kept_idx}]\n{shorten_text(cleaned, max_chars=max_chars)}")
        kept_idx += 1
    return "\n\n".join(rendered)


def pick_shortest(values: Sequence[str]) -> str:
    return min(values, key=lambda item: len(item.strip()))

