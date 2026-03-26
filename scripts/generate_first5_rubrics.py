import json
from pathlib import Path

import pyarrow as pa
import pyarrow.ipc as ipc
from datasets import Dataset


DATASET_DIR = Path("/workspace/action_abstraction/sft_dataset_with_abs_conditioned_solutions_correct")
OUTPUT_DIR = Path("/workspace/action_abstraction/sft_dataset_first5_with_rubrics")
OUTPUT_JSONL = Path("/workspace/action_abstraction/sft_dataset_first5_with_rubrics.jsonl")


RUBRICS = [
    """Checkpoints (max 7 pts total)

- 2 pts: Uses divisibility by 36 as divisibility by both 4 and 9, with the corresponding last-two-digits and digit-sum conditions.
- 1 pt: Shows no 5-digit candidate can work because the only five distinct even digits are 0,2,4,6,8, whose sum is 20, not divisible by 9.
- 1 pt: Identifies the valid 4-digit digit set as 0,4,6,8 (equivalently, removes 2 so the digit sum is 18).
- 1 pt: Maximizes the arrangement as 8640 and verifies divisibility by 4 from the last two digits 40.
- 3 pts: Gives the final answer \\boxed{8640}.

Zero-credit items

- Listing even digits or candidate numbers without connecting them to divisibility by 36.
- Checking only divisibility by 4 or only divisibility by 9.
- Stating that 86420 is large but not explaining why no 5-digit option can work.
- Reordering digits routinely after the correct set is already identified.

Deductions

- –2: Uses an incorrect divisibility criterion for 36, 9, or 4.
- –2: Claims a 5-digit number can work without resolving the digit-sum obstruction.
- –1: Arithmetic slip in a digit sum or divisibility check that does not invalidate the whole method.
- cap at 4/7: Finds a plausible candidate such as 8640 by search but gives no valid justification that it is the largest multiple of 36.
- –1: Final answer not enclosed in \\boxed{}.""",
    """Checkpoints (max 7 pts total)

- 2 pts: For part (1), models the lamp position as a uniform point on a 6-meter segment and derives the condition 2 < x < 4.
- 1 pt: Computes the part (1) probability as favorable length/total length = 2/6 = 1/3.
- 2 pts: For part (2), recognizes that the sum is even iff the two chosen numbers have the same parity.
- 1 pt: Counts favorable pairs correctly, e.g. \\binom{3}{2}+\\binom{3}{2}=6 out of \\binom{6}{2}=15.
- 1 pt: Gives the final answers \\boxed{\\frac{1}{3}} and \\boxed{\\frac{2}{5}}.

Zero-credit items

- Restating that the lamp is on a rope of length 6 without setting up the interval condition.
- Listing sample pairs in part (2) without a complete count.
- Observing there are three even and three odd numbers without using that to compute the probability.
- Writing only one of the two requested answers.

Deductions

- –2: Uses the wrong favorable interval for part (1), such as including endpoints or using length 4 instead of 2.
- –2: In part (2), counts ordered pairs or otherwise uses the wrong denominator.
- –1: Minor arithmetic simplification error, such as leaving 6/15 unsimplified incorrectly.
- cap at 4/7: Solves only one of the two parts correctly.
- –1: Either final answer not enclosed in \\boxed{}.""",
    """Checkpoints (max 7 pts total)

- 2 pts: Places the triangle in coordinates or otherwise determines a correct geometric model, e.g. B=(0,0), C=(14,0), A=(5,12).
- 1 pt: Identifies the foot of the altitude as D=(5,0) or an equivalent description of AD.
- 1 pt: Finds the orthocenter H correctly, e.g. by intersecting AD with another altitude to get H=(5,15/4).
- 3 pts: Computes HD and HA and gives the final ratio \\boxed{5:11}.

Zero-credit items

- Stating only that the triangle is 13-14-15 or quoting its area without connecting to H, D, and A.
- Finding AD alone with no progress toward the orthocenter.
- Drawing or naming altitudes without computing any relevant distances.
- Giving only a decimal approximation of the ratio.

Deductions

- –2: Assigns inconsistent side lengths or places the altitude on the wrong side.
- –2: Uses an incorrect slope or perpendicular-slope relation, leading to the wrong orthocenter.
- –1: Arithmetic error in solving for coordinates or distances after a correct setup.
- cap at 4/7: Finds A and D correctly but never determines H or an equivalent relation for HD and HA.
- –1: Final answer not enclosed in \\boxed{}.""",
    """Checkpoints (max 7 pts total)

- 2 pts: Introduces y=f(a) and solves f(y)=1 correctly, obtaining y\\in\\{2,0,-4\\}.
- 2 pts: Solves the preimage equations f(a)=2, f(a)=0, and f(a)=-4 with the correct domain checks for the piecewise branches.
- 3 pts: Collects all valid real solutions for a and sums them to \\boxed{-\\frac{15}{16}-\\sqrt{5}}.

Zero-credit items

- Solving only one branch of the piecewise function.
- Listing candidate values from quadratics without checking whether they satisfy the required branch domain.
- Treating f(f(a))=1 as if it were simply f(a)=1.
- Expanding or factoring routine quadratics without connecting them to the composition.

Deductions

- –2: Omits the key intermediate set \\{2,0,-4\\} or an equivalent preimage-of-1 step.
- –2: Keeps extraneous roots that violate the branch condition, or discards valid roots for that reason.
- –1: Arithmetic error in summing the valid solutions after the correct set is found.
- cap at 4/7: Correctly finds some but not all solution values of a.
- –1: Final answer not enclosed in \\boxed{}.""",
    """Checkpoints (max 7 pts total)

- 2 pts: Uses the definition of abundant number correctly and checks early integers systematically or cites the first abundant numbers accurately.
- 2 pts: Verifies that 12 and 18 are abundant but are multiples of 6, and that no smaller nonmultiple of 6 qualifies.
- 3 pts: Verifies that 20 is abundant via 1+2+4+5+10=22>20 and gives the final answer \\boxed{20}.

Zero-credit items

- Repeating the definition of abundant number without applying it to any numbers.
- Stating that 12 is abundant only because it is given in the prompt.
- Naming 20 without checking its proper divisors.
- Listing later abundant numbers once 20 has already been justified.

Deductions

- –2: Miscomputes the sum of proper divisors for 12, 18, or 20 in a way that affects the conclusion.
- –2: Fails to justify why no smaller nonmultiple of 6 works.
- –1: Minor omission in the divisor list for 20 if the final inequality is still somehow correct.
- cap at 4/7: Identifies 20 by memory or guess without verification from the definition.
- –1: Final answer not enclosed in \\boxed{}.""",
]


def load_first_five_rows():
    shard = DATASET_DIR / "data-00000-of-00016.arrow"
    with pa.memory_map(str(shard), "r") as source:
        table = ipc.open_stream(source).read_all()
    return table.slice(0, 5).to_pylist()


def main():
    rows = load_first_five_rows()
    for row, rubric in zip(rows, RUBRICS):
        row["rubric"] = rubric

    dataset = Dataset.from_list(rows)

    if OUTPUT_DIR.exists():
        import shutil

        shutil.rmtree(OUTPUT_DIR)
    dataset.save_to_disk(str(OUTPUT_DIR))

    with OUTPUT_JSONL.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Saved dataset to {OUTPUT_DIR}")
    print(f"Saved JSONL to {OUTPUT_JSONL}")
    print(dataset)


if __name__ == "__main__":
    main()
