from math_verify import parse, verify
import re


def extract_answer_from_solution(text):
    m = list(re.finditer(r'\\boxed\s*\{', text))
    if not m:
        return None

    box_start = m[-1].start()   # include "\boxed{"
    start = m[-1].end()         # position right after "{"

    depth = 1
    i = start
    while i < len(text) and depth > 0:
        c = text[i]
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
        i += 1

    if depth != 0:
        return None

    return text[box_start:i]    # includes "\boxed{...}"


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    pred = extract_answer_from_solution(solution_str)
    
    ground_truth_boxed = "\\boxed{" + ground_truth + "}"
    
    if verify(parse(ground_truth_boxed), parse(pred)):
        return 1.0
    else:
        return 0.0
