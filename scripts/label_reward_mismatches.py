import csv
import json
import os

import pyarrow as pa
import pyarrow.parquet as pq


DATA_DIR = "/workspace/action_abstraction/verl_data/sft_dataset_no_rl_partial_05_less1_traintest"
CONCAT_DATA_DIR = (
    "/workspace/action_abstraction/verl_data/sft_dataset_no_rl_partial_05_less1_traintest_concat"
)


# Confirmed label mismatches from manual audit. Each entry is checked against
# both the prompt text and the stored ground truth before labeling.
CONFIRMED_MISMATCHES = {
    "train": {
        3: {
            "ground_truth": "-25",
            "corrected_ground_truth": "-25, 25",
            "problem_substring": "\\cos n^\\circ = \\cos 745^\\circ",
            "reason": "missing_symmetric_solution",
            "note": "Both -25 and 25 satisfy the cosine equation in the stated interval.",
        },
        195: {
            "ground_truth": "18",
            "corrected_ground_truth": "-18, 18",
            "problem_substring": "Find all integers $A$ if it is known that $A^{6}$",
            "reason": "missing_negative_solution",
            "note": "Both -18 and 18 produce the required eight-digit sixth power.",
        },
        410: {
            "ground_truth": "18",
            "corrected_ground_truth": "12, 18",
            "problem_substring": "Enter the two smallest solutions, separated by commas.",
            "reason": "partial_label_last_item_only",
            "note": "Prompt asks for two smallest solutions, but stored label only keeps the second value.",
        },
        745: {
            "ground_truth": "(8,38)",
            "corrected_ground_truth": "(2,-22); (8,38)",
            "problem_substring": "List the points in order of increasing $x$-coordinate, separated by semicolons.",
            "reason": "partial_label_last_item_only",
            "note": "Prompt asks for both intersection points, but stored label only keeps the second point.",
        },
        828: {
            "ground_truth": "-1.41",
            "corrected_ground_truth": "-1.41, 1.41",
            "problem_substring": "has a unique solution. If necessary, round your answer to two decimal places.",
            "reason": "missing_symmetric_solution",
            "note": "The line is tangent to the ellipse for a = +/-sqrt(2), so both rounded values should be present.",
        },
        924: {
            "ground_truth": "-70",
            "corrected_ground_truth": "-70, 70",
            "problem_substring": "\\cos n^\\circ = \\cos 430^\\circ",
            "reason": "missing_symmetric_solution",
            "note": "Both -70 and 70 satisfy the cosine equation in the stated interval.",
        },
        937: {
            "ground_truth": "-2x-1",
            "corrected_ground_truth": "-2x-1, 2x+1",
            "problem_substring": "Enter all possible polynomials $g(x),$ separated by commas.",
            "reason": "partial_label_last_item_only",
            "note": "Prompt asks for all possible polynomials, but stored label only keeps one of the two valid polynomials.",
        },
        1197: {
            "ground_truth": "96",
            "corrected_ground_truth": "-84, 96",
            "problem_substring": "\\tan n^\\circ = \\tan 276^\\circ",
            "reason": "missing_periodic_solution",
            "note": "Both -84 and 96 satisfy the tangent equation in the stated interval.",
        },
        1347: {
            "ground_truth": "-\\frac{5}{2}",
            "corrected_ground_truth": "-\\frac{5}{2}, \\frac{2}{5}",
            "problem_substring": "x-t \\) is a factor of \\( 10x^2 + 21x - 10 \\)",
            "reason": "partial_label_last_item_only",
            "note": "Both roots of the quadratic give valid values of t.",
        },
        1599: {
            "ground_truth": "-135",
            "corrected_ground_truth": "-135, -45",
            "problem_substring": "\\sin m^\\circ = \\sin 945^\\circ",
            "reason": "missing_periodic_solution",
            "note": "Both -135 and -45 satisfy the sine equation in the stated interval.",
        },
        2816: {
            "ground_truth": "60",
            "corrected_ground_truth": "-120, 60",
            "problem_substring": "\\tan n^\\circ = \\tan 1500^\\circ",
            "reason": "missing_periodic_solution",
            "note": "Both -120 and 60 satisfy the tangent equation in the stated interval.",
        },
    },
    "test": {
        40: {
            "ground_truth": "3 - 2\\sqrt{6}",
            "corrected_ground_truth": "3 - 2\\sqrt{6}, 3 + 2\\sqrt{6}",
            "problem_substring": "Find all real numbers $k$ for which there exists a nonzero, 2-dimensional vector",
            "reason": "partial_label_last_item_only",
            "note": "The matrix has two real eigenvalues, 3 - 2sqrt(6) and 3 + 2sqrt(6).",
        },
        86: {
            "ground_truth": "432432",
            "corrected_ground_truth": "392436, 432432",
            "problem_substring": "Find a six-digit number $\\overline{xy243z}$ that is divisible by 396.",
            "reason": "multiple_valid_answers_prompt_ambiguous",
            "note": "Both 392436 and 432432 satisfy the condition, so the prompt is ambiguous while the stored label accepts only one valid answer.",
        },
    },
}


def load_rows(path):
    table = pq.read_table(path)
    return table, table.to_pylist()


def validate_mismatch_row(split, index, row, expected):
    problem = row["problem"].replace("\n", " ")
    gt = row["reward_model"]["ground_truth"]

    if gt != expected["ground_truth"]:
        raise ValueError(
            f"{split}[{index}] ground truth mismatch: expected {expected['ground_truth']!r}, found {gt!r}"
        )

    if expected["problem_substring"] not in problem:
        raise ValueError(
            f"{split}[{index}] problem text no longer matches expected audit target"
        )


def label_rows(split, rows):
    mismatches = CONFIRMED_MISMATCHES[split]
    labeled = []
    mismatch_count = 0
    patch_rows = []

    for index, row in enumerate(rows):
        row = dict(row)

        is_mismatch = index in mismatches
        if is_mismatch:
            validate_mismatch_row(split, index, row, mismatches[index])
            mismatch_count += 1
            row["label_mismatch"] = True
            row["label_mismatch_reason"] = mismatches[index]["reason"]
            row["label_mismatch_note"] = mismatches[index]["note"]
            row["corrected_ground_truth"] = mismatches[index]["corrected_ground_truth"]
            patch_rows.append(
                {
                    "split": split,
                    "index": index,
                    "stored_ground_truth": mismatches[index]["ground_truth"],
                    "corrected_ground_truth": mismatches[index]["corrected_ground_truth"],
                    "reason": mismatches[index]["reason"],
                    "note": mismatches[index]["note"],
                    "problem": row["problem"],
                }
            )
        else:
            row["label_mismatch"] = False
            row["label_mismatch_reason"] = ""
            row["label_mismatch_note"] = ""
            row["corrected_ground_truth"] = ""

        labeled.append(row)

    expected_count = len(mismatches)
    if mismatch_count != expected_count:
        raise ValueError(
            f"{split}: labeled {mismatch_count} mismatches but expected {expected_count}"
        )

    return labeled, patch_rows


def label_concat_rows(rows):
    labeled = []
    mismatch_count = 0
    patch_rows = []
    expected_count = sum(len(mismatches) for mismatches in CONFIRMED_MISMATCHES.values())

    for index, row in enumerate(rows):
        row = dict(row)
        extra_info = row.get("extra_info") or {}
        split = extra_info.get("split")
        split_index = extra_info.get("index")
        mismatch = CONFIRMED_MISMATCHES.get(split, {}).get(split_index)

        if mismatch is not None:
            validate_mismatch_row(split, split_index, row, mismatch)
            mismatch_count += 1
            row["label_mismatch"] = True
            row["label_mismatch_reason"] = mismatch["reason"]
            row["label_mismatch_note"] = mismatch["note"]
            row["corrected_ground_truth"] = mismatch["corrected_ground_truth"]
            patch_rows.append(
                {
                    "split": split,
                    "index": split_index,
                    "concat_index": index,
                    "stored_ground_truth": mismatch["ground_truth"],
                    "corrected_ground_truth": mismatch["corrected_ground_truth"],
                    "reason": mismatch["reason"],
                    "note": mismatch["note"],
                    "problem": row["problem"],
                }
            )
        else:
            row["label_mismatch"] = False
            row["label_mismatch_reason"] = ""
            row["label_mismatch_note"] = ""
            row["corrected_ground_truth"] = ""

        labeled.append(row)

    if mismatch_count != expected_count:
        raise ValueError(
            f"concat: labeled {mismatch_count} mismatches but expected {expected_count}"
        )

    return labeled, patch_rows


def write_table(rows, path):
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path)


def write_patch_csv(rows, path):
    fieldnames = [
        "split",
        "index",
        "concat_index",
        "stored_ground_truth",
        "corrected_ground_truth",
        "reason",
        "note",
        "problem",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_patch_json(rows, path):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2, ensure_ascii=True)


def process_split(split):
    src_path = os.path.join(DATA_DIR, f"{split}.parquet")
    labeled_path = os.path.join(DATA_DIR, f"{split}_mismatch_labeled.parquet")
    filtered_path = os.path.join(DATA_DIR, f"{split}_mismatch_filtered.parquet")
    new_filtered_path = os.path.join(DATA_DIR, f"{split}_new_filtered.parquet")
    patch_csv_path = os.path.join(DATA_DIR, f"{split}_mismatch_patch.csv")
    patch_json_path = os.path.join(DATA_DIR, f"{split}_mismatch_patch.json")

    _, rows = load_rows(src_path)
    labeled_rows, patch_rows = label_rows(split, rows)
    filtered_rows = [row for row in labeled_rows if not row["label_mismatch"]]

    write_table(labeled_rows, labeled_path)
    write_table(filtered_rows, filtered_path)
    write_table(filtered_rows, new_filtered_path)
    write_patch_csv(patch_rows, patch_csv_path)
    write_patch_json(patch_rows, patch_json_path)

    print(
        split,
        {
            "source_rows": len(rows),
            "mismatch_rows": sum(1 for row in labeled_rows if row["label_mismatch"]),
            "filtered_rows": len(filtered_rows),
            "labeled_path": labeled_path,
            "filtered_path": filtered_path,
            "patch_csv_path": patch_csv_path,
            "patch_json_path": patch_json_path,
            "new_filtered_path": new_filtered_path,
        },
    )


def process_concat():
    src_path = os.path.join(CONCAT_DATA_DIR, "train.parquet")
    labeled_path = os.path.join(CONCAT_DATA_DIR, "train_mismatch_labeled.parquet")
    filtered_path = os.path.join(CONCAT_DATA_DIR, "train_mismatch_filtered.parquet")
    new_filtered_path = os.path.join(CONCAT_DATA_DIR, "train_new_filtered.parquet")
    patch_csv_path = os.path.join(CONCAT_DATA_DIR, "train_mismatch_patch.csv")
    patch_json_path = os.path.join(CONCAT_DATA_DIR, "train_mismatch_patch.json")

    _, rows = load_rows(src_path)
    labeled_rows, patch_rows = label_concat_rows(rows)
    filtered_rows = [row for row in labeled_rows if not row["label_mismatch"]]

    write_table(labeled_rows, labeled_path)
    write_table(filtered_rows, filtered_path)
    write_table(filtered_rows, new_filtered_path)
    write_patch_csv(patch_rows, patch_csv_path)
    write_patch_json(patch_rows, patch_json_path)

    print(
        "concat",
        {
            "source_rows": len(rows),
            "mismatch_rows": sum(1 for row in labeled_rows if row["label_mismatch"]),
            "filtered_rows": len(filtered_rows),
            "labeled_path": labeled_path,
            "filtered_path": filtered_path,
            "patch_csv_path": patch_csv_path,
            "patch_json_path": patch_json_path,
            "new_filtered_path": new_filtered_path,
        },
    )


def main():
    for split in ["train", "test"]:
        process_split(split)
    process_concat()


if __name__ == "__main__":
    main()
