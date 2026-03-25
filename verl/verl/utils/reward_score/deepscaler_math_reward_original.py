import re
from typing import Iterable, Optional

import sympy
from pylatexenc import latex2text
from sympy.parsing import sympy_parser

BAD_SUBSTRINGS = ["^{", "^("]
BAD_REGEXES = [r"\^[0-9]+\^", r"\^[0-9][0-9]+"]
TUPLE_CHARS = "()[]"


def extract_answer_from_solution(text: str) -> Optional[str]:
    matches = list(re.finditer(r"\\boxed\s*\{", text))
    if not matches:
        return None

    box_start = matches[-1].start()
    start_brace = matches[-1].end()

    depth = 1
    i = start_brace
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1

    if depth != 0:
        return None

    return text[box_start:i]


def unbox(boxed: str) -> Optional[str]:
    match = re.fullmatch(r"\s*\\boxed\s*\{(.*)\}\s*", boxed, flags=re.DOTALL)
    return match.group(1).strip() if match else None


def mathd_normalize_answer(answer: Optional[str]) -> Optional[str]:
    if answer is None:
        return None

    answer = answer.strip()
    try:
        match = re.search(r"^\\text\{(?P<text>.+?)\}$", answer)
        if match is not None:
            answer = match.group("text").strip()
        return _strip_string(answer)
    except Exception:
        return answer


def _strip_string(text: str) -> str:
    def fix_fracs(value: str) -> str:
        pieces = value.split("\\frac")
        rebuilt = pieces[0]
        if len(pieces) == 1:
            return value

        for piece in pieces[1:]:
            rebuilt += "\\frac"
            if piece and piece[0] == "{":
                rebuilt += piece
                continue

            if len(piece) < 2:
                return value

            left = piece[0]
            right = piece[1]
            tail = piece[2:]
            if right != "{":
                rebuilt += "{" + left + "}{" + right + "}" + tail
            else:
                rebuilt += "{" + left + "}" + right + tail
        return rebuilt

    def fix_simple_fraction(value: str) -> str:
        pieces = value.split("/")
        if len(pieces) != 2:
            return value

        left, right = pieces
        try:
            left_int = int(left)
            right_int = int(right)
        except Exception:
            return value

        if value != f"{left_int}/{right_int}":
            return value
        return f"\\frac{{{left_int}}}{{{right_int}}}"

    def remove_right_units(value: str) -> str:
        if "\\text{ " not in value:
            return value
        pieces = value.split("\\text{ ")
        if len(pieces) != 2:
            return value
        return pieces[0]

    def fix_sqrt(value: str) -> str:
        if "\\sqrt" not in value:
            return value

        pieces = value.split("\\sqrt")
        rebuilt = pieces[0]
        for piece in pieces[1:]:
            if piece and piece[0] != "{":
                rebuilt += "\\sqrt{" + piece[0] + "}" + piece[1:]
            else:
                rebuilt += "\\sqrt" + piece
        return rebuilt

    text = text.replace("\n", "")
    text = text.replace("\\!", "")
    text = text.replace("\\\\", "\\")
    text = text.replace("tfrac", "frac")
    text = text.replace("dfrac", "frac")
    text = text.replace("\\left", "")
    text = text.replace("\\right", "")
    text = text.replace("^{\\circ}", "")
    text = text.replace("^\\circ", "")
    text = text.replace("\\$", "")
    text = remove_right_units(text)
    text = text.replace("\\%", "")
    text = text.replace(" .", " 0.")
    text = text.replace("{.", "{0.")

    if not text:
        return text
    if text[0] == ".":
        text = "0" + text

    pieces = text.split("=")
    if len(pieces) == 2 and len(pieces[0]) <= 2:
        text = pieces[1]

    text = fix_sqrt(text)
    text = text.replace(" ", "")
    text = fix_fracs(text)
    if text == "0.5":
        text = "\\frac{1}{2}"
    text = fix_simple_fraction(text)
    return text


def _parse_latex(expr: str) -> str:
    expr = expr.replace("\\tfrac", "\\frac")
    expr = expr.replace("\\dfrac", "\\frac")
    expr = expr.replace("\\frac", " \\frac")
    expr = latex2text.LatexNodes2Text().latex_to_text(expr)
    expr = expr.replace("√", "sqrt")
    expr = expr.replace("π", "pi")
    expr = expr.replace("∞", "inf")
    expr = expr.replace("∪", "U")
    expr = expr.replace("·", "*")
    expr = expr.replace("×", "*")
    return expr.strip()


def _is_float(text: str) -> bool:
    try:
        float(text)
        return True
    except ValueError:
        return False


def _is_int(value: float) -> bool:
    try:
        return abs(value - int(round(value))) <= 1e-7
    except Exception:
        return False


def _is_frac(expr: str) -> bool:
    return bool(re.search(r"^-?[0-9]+.?/0*[1-9][0-9]*.?$", expr))


def _strip_properly_formatted_commas(expr: str) -> str:
    pattern = re.compile(r"(\d)(,)(\d\d\d)($|\D)")
    while True:
        updated = pattern.sub(r"\1\3\4", expr)
        if updated == expr:
            return updated
        expr = updated


def _str_is_int(expr: str) -> bool:
    try:
        expr = _strip_properly_formatted_commas(expr)
        value = float(expr)
        return abs(value - int(round(value))) <= 1e-7
    except Exception:
        return False


def _str_to_int(expr: str) -> int:
    return int(float(expr.replace(",", "")))


def _inject_implicit_mixed_number(expr: str) -> str:
    return re.compile(r"([0-9]) +([0-9])").sub(r"\1+\2", expr)


def _normalize(expr: Optional[str]) -> Optional[str]:
    if expr is None:
        return None

    match = re.search(r"^\\text\{(?P<text>.+?)\}$", expr)
    if match is not None:
        expr = match.group("text")

    expr = expr.replace("\\%", "%")
    expr = expr.replace("\\$", "$")
    expr = expr.replace("$", "")
    expr = expr.replace("%", "")
    expr = expr.replace(" or ", " , ")
    expr = expr.replace(" and ", " , ")
    expr = expr.replace("million", "*10^6")
    expr = expr.replace("billion", "*10^9")
    expr = expr.replace("trillion", "*10^12")

    for unit in [
        "degree",
        "cm",
        "centimeter",
        "meter",
        "mile",
        "second",
        "minute",
        "hour",
        "day",
        "week",
        "month",
        "year",
        "foot",
        "feet",
        "inch",
        "yard",
    ]:
        expr = re.sub(rf"{unit}(es)?(s)? *(\^[0-9]+)?", "", expr)
    expr = re.sub(r"\^ *\\circ", "", expr)

    if len(expr) > 1 and expr[0] == "{" and expr[-1] == "}":
        expr = expr[1:-1]

    expr = re.sub(r",\\! *", "", expr)
    if _is_float(expr) and _is_int(float(expr)):
        expr = str(int(round(float(expr))))

    if "\\" in expr:
        try:
            expr = _parse_latex(expr)
        except Exception:
            pass

    expr = re.sub(r"- *", "-", expr)
    expr = _inject_implicit_mixed_number(expr)
    expr = expr.replace(" ", "")
    expr = expr.replace("{", "")
    expr = expr.replace("}", "")
    expr = expr.lower()

    if _str_is_int(expr):
        expr = str(_str_to_int(expr))

    return expr


def _sympy_parse(expr: str):
    return sympy_parser.parse_expr(
        expr.replace("^", "**"),
        transformations=(
            sympy_parser.standard_transformations
            + (sympy_parser.implicit_multiplication_application,)
        ),
    )


def _count_unknown_letters(expr: str) -> int:
    expr = expr.replace("sqrt", "")
    expr = expr.replace("frac", "")
    return len({char for char in expr if char.isalpha()})


def _should_allow_eval(expr: str) -> bool:
    if _count_unknown_letters(expr) > 2:
        return False
    if any(bad in expr for bad in BAD_SUBSTRINGS):
        return False
    return not any(re.search(pattern, expr) for pattern in BAD_REGEXES)


def _are_equal_under_sympy(left: str, right: str) -> bool:
    try:
        expr = f"({left})-({right})"
        if not _should_allow_eval(expr):
            return False
        return sympy.simplify(_sympy_parse(expr)) == 0
    except Exception:
        return False


def _split_tuple(expr: str) -> list[str]:
    expr = _strip_properly_formatted_commas(expr)
    if not expr:
        return []

    if (
        len(expr) > 2
        and expr[0] in TUPLE_CHARS
        and expr[-1] in TUPLE_CHARS
        and all(ch not in expr[1:-1] for ch in TUPLE_CHARS)
    ):
        return [part.strip() for part in expr[1:-1].split(",")]
    return [expr]


def grade_answer_mathd(given_answer: str, ground_truth: str) -> bool:
    return mathd_normalize_answer(given_answer) == mathd_normalize_answer(ground_truth)


def grade_answer_sympy(given_answer: str, ground_truth: str) -> bool:
    gold = _normalize(ground_truth)
    pred = _normalize(given_answer)

    if gold is None or pred is None:
        return False
    if gold == pred:
        return True
    if not pred:
        return False

    gold_parts = _split_tuple(gold)
    pred_parts = _split_tuple(pred)

    if len(gold_parts) > 1 and (gold[0] != pred[0] or gold[-1] != pred[-1]):
        return False
    if len(gold_parts) != len(pred_parts):
        return False

    for gold_part, pred_part in zip(gold_parts, pred_parts):
        if _is_frac(gold_part) and _is_frac(pred_part):
            if gold_part != pred_part:
                return False
            continue

        if _str_is_int(gold_part) != _str_is_int(pred_part):
            return False

        if not _are_equal_under_sympy(gold_part, pred_part):
            return False

    return True


def _iter_ground_truths(ground_truth) -> Iterable[str]:
    if isinstance(ground_truth, (list, tuple, set)):
        values = ground_truth
    else:
        values = [ground_truth]

    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if not text:
            continue

        if "\\boxed" in text:
            boxed = extract_answer_from_solution(text)
            unboxed = unbox(boxed) if boxed is not None else None
            if unboxed:
                yield unboxed
        else:
            yield text


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    # del data_source, extra_info

    pred_boxed = extract_answer_from_solution(solution_str)
    if pred_boxed is None:
        return 0.0

    pred_answer = unbox(pred_boxed)
    if pred_answer is None:
        return 0.0

    for candidate in _iter_ground_truths(ground_truth):
        if grade_answer_mathd(pred_answer, candidate) or grade_answer_sympy(pred_answer, candidate):
            return 1.0

    return 0.0
