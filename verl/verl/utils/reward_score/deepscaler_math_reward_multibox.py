from math_verify import parse, verify
import re


# Canonical 24-hour times like 6:00, 11:00, 14:50, 19:59.
TIME_24_RE = re.compile(r"^\s*(?:[0-9]|1[0-9]|2[0-3]):[0-5][0-9]\s*$")
# 12-hour times with minutes like 2:50 PM, 02:50pm.
TIME_12_MIN_RE = re.compile(r"^\s*(1[0-2]|0?[1-9]):([0-5][0-9])\s*([AaPp][Mm])\s*$")
# 12-hour times without minutes like 11 AM, 7pm.
TIME_12_HOUR_RE = re.compile(r"^\s*(1[0-2]|0?[1-9])\s*([AaPp][Mm])\s*$")


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


def unique_in_order(values):
    seen = set()
    out = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def build_candidate_boxed_predictions(solution_str, gt_raw):
    boxed_answers = extract_boxed_answers(solution_str)
    if not boxed_answers:
        return []

    candidates = list(boxed_answers)
    raw_answers = [unbox(boxed) for boxed in boxed_answers]

    # If the gold answer is multi-part, also score common combined renderings
    # produced when the model boxes each final item separately.
    joiners = []
    if ";" in gt_raw:
        joiners.extend(["; ", ";"])
    if "," in gt_raw:
        joiners.extend([", ", ","])

    # Fall back to a few generic joiners for multi-box outputs even when the
    # gold format is ambiguous.
    if len(raw_answers) > 1:
        joiners.extend([", ", ",", "; ", ";"])

    for joiner in unique_in_order(joiners):
        candidates.append("\\boxed{" + joiner.join(raw_answers) + "}")

    return unique_in_order(candidates)


def score_boxed_candidate(pred_boxed, gt_raw):
    pred_raw = unbox(pred_boxed)

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


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    gt_raw = ground_truth.strip()
    candidates = build_candidate_boxed_predictions(solution_str, gt_raw)
    if not candidates:
        return 0.0

    return max(score_boxed_candidate(candidate, gt_raw) for candidate in candidates)
