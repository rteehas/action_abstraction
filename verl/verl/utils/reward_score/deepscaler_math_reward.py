from math_verify import parse, verify
import re

def extract_answer_from_solution(text):
    m = list(re.finditer(r'\\boxed\s*\{', text))
    if not m:
        return None

    box_start = m[-1].start()
    start = m[-1].end()

    depth = 1
    i = start
    while i < len(text) and depth > 0:
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
        i += 1

    if depth != 0:
        return None

    return text[box_start:i]

# def compute_score(data_source, solution_str, ground_truth, extra_info=None):
#     pred = extract_answer_from_solution(solution_str)
#     if pred is None:
#         return 0.0

#     gold = f"\\boxed{{{ground_truth}}}"

#     try:
#         gold_parsed = parse(
#             gold,
#             parsing_timeout=None,
#             raise_on_error=True,
#         )
#         pred_parsed = parse(
#             pred,
#             parsing_timeout=None,
#             raise_on_error=True,
#         )

#         ok = verify(
#             gold_parsed,
#             pred_parsed,
#             timeout_seconds=None,
#             raise_on_error=True,
#         )
#         return 1.0 if ok else 0.0

#     except Exception as e:
#         print("SCORING ERROR")
#         print("gold:", gold)
#         print("pred:", pred)
#         print("err:", repr(e))
#         return 0.0

TIME_RE = re.compile(r"^\s*(\d{1,2})\s*:\s*(\d{2})\s*$")
MCQ_RE = re.compile(r"^\s*[A-Z]\s*$")
PLAIN_STRING_RE = re.compile(r"^[A-Za-z][A-Za-z0-9 _-]*$")

def extract_answer_from_solution(text):
    m = list(re.finditer(r'\\boxed\s*\{', text))
    if not m:
        return None

    start_brace = m[-1].end()
    box_start = m[-1].start()

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

def unbox(boxed):
    m = re.fullmatch(r"\s*\\boxed\s*\{(.*)\}\s*", boxed, flags=re.DOTALL)
    return m.group(1).strip() if m else boxed.strip()

def normalize_string_answer(s):
    s = s.strip()
    m = TIME_RE.fullmatch(s)
    if m:
        return f"{int(m.group(1))}:{m.group(2)}"
    return re.sub(r"\s+", " ", s)

def gold_answer_type(gt):
    gt = gt.strip()

    if TIME_RE.fullmatch(gt):
        return "string"

    if MCQ_RE.fullmatch(gt):
        return "string"

    if PLAIN_STRING_RE.fullmatch(gt) and not re.search(r"[=+\\^*/{}[\]()]", gt):
        return "string"

    return "math"

def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    pred_boxed = extract_answer_from_solution(solution_str)
    if pred_boxed is None:
        return 0.0

    pred_raw = unbox(pred_boxed)
    gt_raw = ground_truth.strip()

    answer_type = gold_answer_type(gt_raw)

    if answer_type == "string":
        return 1.0 if normalize_string_answer(pred_raw) == normalize_string_answer(gt_raw) else 0.0

    gold_boxed = f"\\boxed{{{gt_raw}}}"

    try:
        gold_parsed = parse(
            gold_boxed,
            parsing_timeout=None,
            raise_on_error=True,
        )
        pred_parsed = parse(
            pred_boxed,
            parsing_timeout=None,
            raise_on_error=True,
        )
        ok = verify(
            gold_parsed,
            pred_parsed,
            timeout_seconds=None,
            raise_on_error=True,
        )
        return 1.0 if ok else 0.1
    except Exception as e:
        print("SCORING ERROR")
        print("gold:", gold_boxed)
        print("pred:", pred_boxed)
        print("err:", repr(e))
        return 0.1