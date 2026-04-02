import re
from math_verify import parse, verify


# Canonical 24-hour times like 6:00, 11:00, 14:50, 19:59.
TIME_24_RE = re.compile(r"^\s*(?:[0-9]|1[0-9]|2[0-3]):[0-5][0-9]\s*$")
# 12-hour times with minutes like 2:50 PM, 02:50pm.
TIME_12_MIN_RE = re.compile(r"^\s*(1[0-2]|0?[1-9]):([0-5][0-9])\s*([AaPp][Mm])\s*$")
# 12-hour times without minutes like 11 AM, 7pm.
TIME_12_HOUR_RE = re.compile(r"^\s*(1[0-2]|0?[1-9])\s*([AaPp][Mm])\s*$")
PERCENT_WORD_RE = re.compile(r"\bpercent(?:age)?s?\b", flags=re.IGNORECASE)


def extract_boxed_answers(text):
    matches = list(re.finditer(r"\\boxed\s*\{", text))
    if not matches:
        return []

    boxed = []
    for match in matches:
        box_start = match.start()
        start_brace = match.end()

        depth = 1
        i = start_brace
        while i < len(text) and depth > 0:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1

        if depth == 0:
            boxed.append(text[box_start:i])

    return boxed


def unbox(boxed):
    match = re.fullmatch(r"\s*\\boxed\s*\{(.*)\}\s*", boxed, flags=re.DOTALL)
    return match.group(1).strip() if match else boxed.strip()


def _cleanup_time_text(s):
    s = s.strip()
    s = s.strip("$")
    s = s.replace("\\!", "")
    s = s.replace("\\,", " ")
    s = s.replace("\\;", " ")
    s = s.replace("\\:", ":")
    s = s.replace("\\.", ".")

    # Common LaTeX wrappers around AM/PM.
    s = re.sub(r"\\text\s*\{\s*([AaPp][Mm])\s*\}", r" \1", s)
    s = re.sub(r"\\mathrm\s*\{\s*([AaPp][Mm])\s*\}", r" \1", s)

    s = re.sub(r"\s+", " ", s).strip()
    s = s.rstrip(".")
    return s


def _to_24_hour(hour, minute, meridiem):
    hour = int(hour)
    meridiem = meridiem.upper()
    if meridiem == "AM":
        hour = 0 if hour == 12 else hour
    else:
        hour = 12 if hour == 12 else hour + 12
    return f"{hour}:{minute}"


def normalize_time_answer(s):
    s = _cleanup_time_text(s)

    match = TIME_24_RE.fullmatch(s)
    if match:
        hour, minute = s.split(":")
        return f"{int(hour)}:{minute}"

    match = TIME_12_MIN_RE.fullmatch(s)
    if match:
        hour, minute, meridiem = match.groups()
        return _to_24_hour(hour, minute, meridiem)

    match = TIME_12_HOUR_RE.fullmatch(s)
    if match:
        hour, meridiem = match.groups()
        return _to_24_hour(hour, "00", meridiem)

    return None


def gt_is_percent_like(text):
    if not text:
        return False
    return ("%" in text) or (r"\%" in text) or bool(PERCENT_WORD_RE.search(text))


def strip_percent_markers(text):
    text = text.strip()
    text = text.replace(r"\%", "")
    text = text.replace("%", "")
    text = PERCENT_WORD_RE.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def _strip_outer_wrappers(s):
    s = s.strip()
    pairs = {
        "(": ")",
        "[": "]",
        "{": "}",
    }
    while len(s) >= 2 and s[0] in pairs and s[-1] == pairs[s[0]]:
        s = s[1:-1].strip()
    return s


def _split_top_level(text, delimiter):
    parts = []
    start = 0
    depth_paren = 0
    depth_brack = 0
    depth_brace = 0

    for i, ch in enumerate(text):
        if ch == "(":
            depth_paren += 1
        elif ch == ")":
            depth_paren = max(0, depth_paren - 1)
        elif ch == "[":
            depth_brack += 1
        elif ch == "]":
            depth_brack = max(0, depth_brack - 1)
        elif ch == "{":
            depth_brace += 1
        elif ch == "}":
            depth_brace = max(0, depth_brace - 1)
        elif (
            ch == delimiter
            and depth_paren == 0
            and depth_brack == 0
            and depth_brace == 0
        ):
            parts.append(text[start:i].strip())
            start = i + 1

    parts.append(text[start:].strip())
    return parts


def normalize_multipart_answer(s, delimiter):
    s = _strip_outer_wrappers(s.strip())
    parts = [part for part in _split_top_level(s, delimiter) if part]
    if len(parts) <= 1:
        return None
    return parts


def score_multipart_answer(pred_raw, gt_raw):
    for delimiter in (",", ";"):
        if delimiter not in gt_raw:
            continue

        gt_parts = normalize_multipart_answer(gt_raw, delimiter)
        pred_parts = normalize_multipart_answer(pred_raw, delimiter)
        if gt_parts is None or pred_parts is None:
            continue
        if len(gt_parts) != len(pred_parts):
            return 0.0

        for gt_part, pred_part in zip(gt_parts, pred_parts):
            gt_boxed = f"\\boxed{{{gt_part}}}"
            pred_boxed = f"\\boxed{{{pred_part}}}"
            if score_single_answer(pred_boxed, gt_part) != 1.0:
                return 0.0
        return 1.0

    return None


def score_single_answer(pred_boxed, gt_raw):
    pred_raw = unbox(pred_boxed)

    multipart_score = score_multipart_answer(pred_raw, gt_raw)
    if multipart_score is not None:
        return multipart_score

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
        return 1.0 if ok else 0.0
    except Exception as e:
        # Last-resort exact match for parser edge cases without broadly
        # reclassifying symbolic math expressions as strings.
        if pred_raw.strip() == gt_raw:
            return 1.0

        print("SCORING ERROR")
        print("gold:", gold_boxed)
        print("pred:", pred_boxed)
        print("err:", repr(e))
        return 0.0


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    gt_raw = ground_truth.strip()
    boxed_answers = extract_boxed_answers(solution_str)
    if not boxed_answers:
        return 0.0

    # Score only the final boxed answer so the reward matches the model's
    # committed final answer instead of any earlier boxed guess.
    final_boxed = boxed_answers[-1]
    base_score = score_single_answer(final_boxed, gt_raw)
    if base_score == 1.0:
        return 1.0

    if not gt_is_percent_like(gt_raw):
        return base_score

    pred_raw = unbox(final_boxed)
    pred_norm = strip_percent_markers(pred_raw)
    gt_norm = strip_percent_markers(gt_raw)
    if not pred_norm or not gt_norm:
        return base_score

    pred_boxed = f"\\boxed{{{pred_norm}}}"
    return score_single_answer(pred_boxed, gt_norm)
