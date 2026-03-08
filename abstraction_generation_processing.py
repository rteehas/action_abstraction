import re

def extract_abstraction_from_tag(text):

    m = re.search(r'<abstraction>([\s\S]*?)</abstraction>', text)
    inside = m.group(1) if m else None

    return inside

def extract_procedure_from_tag(text):

    m = re.search(r'<procedure>([\s\S]*?)</procedure>', text)
    inside = m.group(1) if m else None

    return inside

def get_abstraction(ex):
    abs_generation = ex["abstraction_generation"]
    ex["abstraction"] = extract_abstraction_from_tag(abs_generation)
    return ex

def get_procedure(ex):
    abs_generation = ex["abstraction_generation"]
    ex["procedure"] = extract_procedure_from_tag(abs_generation)
    return ex

def extract_gt_answer_aime(text: str) -> float:
    """For preprocessing aime dataset ground truth answer"""
    numbers_only = re.sub(r'[^\d.]', '', text) # only digits
    if numbers_only == '':
        return None
    else:
        return float(numbers_only)


