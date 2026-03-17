import re
from sympy import simplify, nsimplify
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations, implicit_multiplication_application
)
from math_verify import parse, verify


TRANSFORMS = standard_transformations + (implicit_multiplication_application,)

def _strip_latex_wrappers(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\\boxed\s*\{([^}]*)\}", r"\1", s)
    s = re.sub(r"^\$|^\$\$|\$$|\$\$$", "", s)
    s = s.replace(r"\left", "").replace(r"\right", "")
    return s

def _latex_frac_to_parens(s: str) -> str:
    return re.sub(r"\\d?frac\s*\{([^}]*)\}\s*\{([^}]*)\}", r"(\1)/(\2)", s)

def _latex_sqrt_to_sympy(s: str) -> str:
    # \sqrt{a} -> sqrt(a)
    s = re.sub(r"\\sqrt\s*\{([^}]*)\}", r"sqrt(\1)", s)
    # \sqrt a (rare) -> sqrt(a) for simple tokens
    s = re.sub(r"\\sqrt\s*([A-Za-z0-9]+)", r"sqrt(\1)", s)
    return s

def _mixed_number_to_sum(s: str) -> str:
    pat = re.compile(r"(?<![A-Za-z0-9_])([+-]?\d+)\s*\\d?frac\s*\{([^}]*)\}\s*\{([^}]*)\}")
    def repl(m):
        whole, num, den = m.group(1), m.group(2), m.group(3)
        if whole.startswith("-"):
            w = whole[1:]
            return f"-({w} + ({num})/({den}))"
        return f"({whole} + ({num})/({den}))"
    return pat.sub(repl, s)

def _basic_rewrites(s: str) -> str:
    s = s.replace("^", "**")
    s = s.replace(r"\cdot", "*").replace(r"\times", "*")
    s = s.replace("−", "-")
    s = re.sub(r"\^\s*\{\\circ\}|\^\s*\\circ|\\circ", " DEG", s)
    s = s.replace(r"\%", " PCT")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _brace_to_paren(s: str) -> str:
    # do this late, after \frac / \sqrt handling
    return s.replace("{", "(").replace("}", ")")

def _equation_rhs_if_trivial(s: str):
    # returns RHS if string is "var = expr" with var being a single symbol/latex symbol
    if "=" not in s:
        return None
    left, right = [t.strip() for t in s.split("=", 1)]
    # accept single ascii letter or a simple latex symbol like \theta
    if re.fullmatch(r"[A-Za-z]", left) or re.fullmatch(r"\\[A-Za-z]+", left):
        return right
    return None

def _to_sympy_candidates(s: str):
    s0 = _strip_latex_wrappers(s)

    # if it's a trivial "y = ..." wrapper, add RHS as an extra candidate seed
    seeds = {s0}
    rhs = _equation_rhs_if_trivial(s0)
    if rhs is not None:
        seeds.add(rhs)

    exprs = []
    for seed in seeds:
        t = _mixed_number_to_sum(seed)
        t = _latex_frac_to_parens(t)
        t = _latex_sqrt_to_sympy(t)
        t = _basic_rewrites(t)

        # unit variants
        variants = set()
        variants.add(_brace_to_paren(t).replace(" DEG", "").replace(" PCT", ""))

        if "PCT" in t:
            variants.add(_brace_to_paren(re.sub(r"(\S+)\s*PCT", r"(\1)/100", t)).replace(" DEG", ""))
            variants.add(_brace_to_paren(t.replace(" DEG", "").replace(" PCT", "")) + "/100")

        for v in variants:
            try:
                exprs.append(parse_expr(v, transformations=TRANSFORMS, evaluate=True))
            except Exception:
                pass
    return exprs

def answers_equivalent(gen: str, gt: str, tol=1e-9) -> bool:
    gen_exprs = _to_sympy_candidates(gen)
    gt_exprs  = _to_sympy_candidates(gt)

    for a in gen_exprs:
        for b in gt_exprs:
            try:
                if simplify(a - b) == 0:
                    return True
            except Exception:
                pass

    for a in gen_exprs:
        for b in gt_exprs:
            try:
                if simplify(nsimplify(a) - nsimplify(b)) == 0:
                    return True
            except Exception:
                pass
            try:
                if abs(float(a.evalf()) - float(b.evalf())) <= tol:
                    return True
            except Exception:
                pass

    return False

# def extract_answer_from_solution(text):
#     m = list(re.finditer(r'\\boxed\{', text))
#     if not m:
#         return None
#     start = m[-1].end()
#     depth = 1
#     i = start
#     while i < len(text) and depth > 0:
#         c = text[i]
#         if c == '{':
#             depth += 1
#         elif c == '}':
#             depth -= 1
#         i += 1
#     if depth != 0:
#         return None
#     return text[start:i-1]

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


def get_generated_answers(ex, answer_column="generated_solution"):
    answers = []
    for soln in ex[answer_column]:
        answers.append(extract_answer_from_solution(soln))
    
    ex["generated_answer"] = answers
    return ex

def sanitize_answer(text):
    text = text.replace("\\dfrac", "\\frac")
    return text 

def compute_num_correct(ex, prefix="", answer_col="answer"):
    gt_answer = "\\boxed{" + ex[answer_col] + "}"
    gt_answer = parse(gt_answer)
    generated_answers = []
    for ans in ex["generated_answer"]:
        if ans is not None:
            generated_answers.append(parse(ans))
        else:
            generated_answers.append(ans)

    correct = 0
    for gen_ans in generated_answers:
        if gen_ans is None:
            continue

        # if (sanitize_answer(gen_ans) == gt_answer) or answers_equivalent(gen_ans, gt_answer):
        #     correct += 1
        if verify(gt_answer, gen_ans):
            correct += 1
    if prefix != "":
        ex[f"{prefix}_num_correct"] = correct
        ex[f"{prefix}_passrate"] = correct / len(generated_answers)
    else:
        ex["num_correct"] = correct
        ex["passrate"] = correct / len(generated_answers)
    return ex

def get_correct_solutions(ex, prefix="", answer_col="answer", solution_column="generated_solution"):
    gt_answer = "\\boxed{" + ex[answer_col] + "}"
    gt_answer = parse(gt_answer)
    generated_answers = []
    for ans in ex["generated_answer"]:
        if ans is not None:
            generated_answers.append(parse(ans))
        else:
            generated_answers.append(ans)

    correct_solutions = []
    for i, gen_ans in enumerate(generated_answers):
        if gen_ans is None:
            continue

        # if (sanitize_answer(gen_ans) == gt_answer) or answers_equivalent(gen_ans, gt_answer):
        #     correct += 1
        if verify(gt_answer, gen_ans):
            correct_solutions.append(ex[solution_column][i])
    
    ex["correct_solutions"] = correct_solutions
    return ex
