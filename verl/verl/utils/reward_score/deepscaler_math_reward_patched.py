from math_verify import parse, verify
import re


# Match valid 24-hour clock times like 6:00, 11:00, 14:50, 19:59.
# This intentionally does not match ratio-like answers such as 1 : 14.
TIME_RE = re.compile(r"^\s*(?:[0-9]|1[0-9]|2[0-3]):[0-5][0-9]\s*$")


def extract_answer_from_solution(text):
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


def unbox(boxed):
    match = re.fullmatch(r"\s*\\boxed\s*\{(.*)\}\s*", boxed, flags=re.DOTALL)
    return match.group(1).strip() if match else boxed.strip()


def normalize_time_answer(s):
    s = s.strip()
    match = TIME_RE.fullmatch(s)
    if not match:
        return None

    hour, minute = s.split(":")
    return f"{int(hour)}:{minute}"


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    pred_boxed = extract_answer_from_solution(solution_str)
    if pred_boxed is None:
        return 0.0

    pred_raw = unbox(pred_boxed)
    gt_raw = ground_truth.strip()

    # Keep exact string matching only for strict clock-time answers.
    pred_time = normalize_time_answer(pred_raw)
    gt_time = normalize_time_answer(gt_raw)
    if gt_time is not None:
        return 1.0 if pred_time == gt_time else 0.0

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
        # Last-resort exact match for parser edge cases without broadly
        # reclassifying symbolic math expressions as strings.
        if pred_raw.strip() == gt_raw:
            return 1.0

        print("SCORING ERROR")
        print("gold:", gold_boxed)
        print("pred:", pred_boxed)
        print("err:", repr(e))
        return 0.1
