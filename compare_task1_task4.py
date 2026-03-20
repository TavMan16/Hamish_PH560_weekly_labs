"""Compare deterministic Task 1 results against stochastic Task 4 results."""

import csv
from pathlib import Path


# === FILE PATHS ===
TASK1_INPUT = Path("task1_summary_N101.csv")
TASK4_INPUT = Path("task4_summary_N101_w10000000x.csv")

COMPARISON_OUTPUT = Path("task5_comparison.csv")
SUMMARY_OUTPUT = Path("task5_summary.csv")


# === MATCH KEYS ===
MATCH_KEYS = (
    "case",
    "start_row",
    "start_column",
    "boundary_name",
    "charge_name",
)


def to_float(x):
    return float(x)


def load_csv(path):
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_index(rows):
    index = {}
    for row in rows:
        key = tuple(row[k] for k in MATCH_KEYS)
        index[key] = row
    return index


def compare_rows(task1, task4):
    t1 = to_float(task1["total_potential"])
    t4 = to_float(task4["total_potential"])
    err = to_float(task4["total_error"])

    diff = t4 - t1
    abs_diff = abs(diff)
    within = abs_diff <= err

    return {
        "case": task4["case"],
        "x_cm": task4["x_cm"],
        "y_cm": task4["y_cm"],
        "boundary_name": task4["boundary_name"],
        "charge_name": task4["charge_name"],
        "task1_total": t1,
        "task4_total": t4,
        "task4_error": err,
        "difference": diff,
        "abs_difference": abs_diff,
        "within_error": within,
    }


def write_csv(path, rows):
    if not rows:
        return

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    print("Loading CSV files...")

    task1_rows = load_csv(TASK1_INPUT)
    task4_rows = load_csv(TASK4_INPUT)

    task1_index = build_index(task1_rows)

    comparison = []
    missing = 0

    for row4 in task4_rows:
        key = tuple(row4[k] for k in MATCH_KEYS)

        if key not in task1_index:
            print(f"WARNING: missing Task 1 case {key}")
            missing += 1
            continue

        row1 = task1_index[key]
        comparison.append(compare_rows(row1, row4))

    print(f"Matched rows: {len(comparison)}")
    print(f"Missing rows: {missing}")

    write_csv(COMPARISON_OUTPUT, comparison)

    # === SUMMARY ===
    within = sum(1 for r in comparison if r["within_error"])
    total = len(comparison)

    summary = [{
        "total_cases": total,
        "within_error": within,
        "fraction_within": within / total if total else 0.0,
    }]

    write_csv(SUMMARY_OUTPUT, summary)

    print(f"Saved: {COMPARISON_OUTPUT}")
    print(f"Saved: {SUMMARY_OUTPUT}")


if __name__ == "__main__":
    main()
