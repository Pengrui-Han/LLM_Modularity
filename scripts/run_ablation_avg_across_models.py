"""
Average ablation matrices across models.

For each (source_task, target_task) cell, averages over all models that have
both tasks present. Missing cells (model hasn't run that pair) are skipped.

Mirrors run_overlap_avg_across_models.py but for ablation_analysis CSVs.

Usage:
    python scripts/run_ablation_avg_across_models.py
    python scripts/run_ablation_avg_across_models.py --pct 1.0
    python scripts/run_ablation_avg_across_models.py --variant row_normalized
"""

import argparse
import os
import sys
import json
import csv
import numpy as np


ALLOWED_MODELS = {
    "Qwen_Qwen2-5-32B-Instruct",
    "Qwen_Qwen2-5-72B-Instruct",
    "allenai_OLMo-2-0325-32B-Instruct",
    "meta-llama_Meta-Llama-3-1-70B-Instruct",
    "mistralai_Mistral-Large-Instruct-2407",
    "mistralai_Mistral-Small-24B-Instruct-2501",
}


def load_ablation_csv(csv_path):
    """Load ablation CSV. First column header is 'source\\target'.
    Returns (matrix, row_labels, col_labels)."""
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        col_labels = header[1:]
        row_labels = []
        matrix = []
        for row in reader:
            row_labels.append(row[0])
            vals = []
            for x in row[1:]:
                x = x.strip()
                if x == "" or x.lower() == "nan":
                    vals.append(np.nan)
                else:
                    vals.append(float(x))
            matrix.append(vals)
    return np.array(matrix), row_labels, col_labels


# ============================================================
# Tasks to exclude from ablation analysis (kept in sync with run_overlap.py / run_ablation_analysis.py)
# ============================================================
EXCLUDE_TASKS = {
    'ToM/r_agent',
    'ToM/r_social_interactions',
    'ToM/r_social_relations',
    'ToM/mixed_social_relations',
    'ToM/mixed_agent',
    'ToM/mixed_social_interactions',
    'phys/mixed_spatial',
    'phys/physical_material_behav',
    'phys/physical_obj_motion_force',
    'phys/physical_spatial_relational',
    'phys/material',
    'phys/spatial',
    'ToM/socialqa',
}


def domain_sort_key(label):
    domain_order = {"MD": 0, "ToM": 1, "phys": 2, "Lan": 3}
    domain = label.split("/")[0]
    task = label.split("/", 1)[1] if "/" in label else ""
    return (domain_order.get(domain, 99), task)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pct", type=str, default="0.1",
                        help="Percentage string matching the CSV filename (e.g. 0.1 or 1.0)")
    parser.add_argument("--variant", default="corrupted_accuracy",
                        choices=["corrupted_accuracy", "corrupted_accuracy_row_normalized",
                                 "accuracy"],
                        help="Which ablation CSV variant to average")
    parser.add_argument("--sign", default="positive")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    filename = f"{args.sign}_{args.pct}pct_{args.variant}.csv"
    print(f"Averaging: {filename}\n")

    # Find per-model CSVs
    found = []
    for model_dir_name in sorted(os.listdir(args.results_dir)):
        if model_dir_name not in ALLOWED_MODELS:
            continue
        csv_path = os.path.join(args.results_dir, model_dir_name,
                                "ablation_analysis", filename)
        if os.path.exists(csv_path):
            found.append((model_dir_name, csv_path))
        else:
            print(f"  (missing) {model_dir_name}")

    if not found:
        print("No CSVs found.")
        sys.exit(1)

    print(f"\nFound {len(found)} models with {filename}:")
    for name, path in found:
        print(f"  {name}")

    # Load all matrices (row/col labels assumed to match per model, but we handle
    # differences via task union)
    model_data = []  # (name, matrix, row_labels, col_labels)
    for name, path in found:
        matrix, rows, cols = load_ablation_csv(path)
        model_data.append((name, matrix, rows, cols))
        print(f"  {name}: {matrix.shape} (rows={len(rows)}, cols={len(cols)})")

    # Task union (rows + cols treated as same task set — they are in this project)
    all_tasks = set()
    for _, _, rows, cols in model_data:
        all_tasks.update(rows)
        all_tasks.update(cols)
    all_tasks = all_tasks - EXCLUDE_TASKS
    all_tasks = sorted(all_tasks, key=domain_sort_key)
    n = len(all_tasks)
    task_to_idx = {t: i for i, t in enumerate(all_tasks)}

    print(f"\nTask union: {n} tasks")

    # Accumulate sum + count per cell, skipping NaN cells
    sum_matrix = np.zeros((n, n))
    count_matrix = np.zeros((n, n), dtype=int)

    for name, matrix, rows, cols in model_data:
        row_idx = {r: i for i, r in enumerate(rows)}
        col_idx = {c: j for j, c in enumerate(cols)}
        for ti in rows:
            if ti not in task_to_idx:
                continue
            for tj in cols:
                if tj not in task_to_idx:
                    continue
                v = matrix[row_idx[ti], col_idx[tj]]
                if np.isnan(v):
                    continue
                gi = task_to_idx[ti]
                gj = task_to_idx[tj]
                sum_matrix[gi, gj] += v
                count_matrix[gi, gj] += 1

    # Average
    avg_matrix = np.where(count_matrix > 0,
                          sum_matrix / np.maximum(count_matrix, 1),
                          np.nan)

    # Output
    out_dir = os.path.join(args.results_dir, "average", "ablation")
    os.makedirs(out_dir, exist_ok=True)

    base = f"{args.sign}_{args.pct}pct_{args.variant}"
    csv_path = os.path.join(out_dir, f"{base}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["source\\target"] + all_tasks)
        for i, ti in enumerate(all_tasks):
            row = [ti]
            for j in range(n):
                v = avg_matrix[i, j]
                row.append("" if np.isnan(v) else f"{v:.6f}")
            writer.writerow(row)
    print(f"\n  Saved: {csv_path}")

    json_path = os.path.join(out_dir, f"{base}.json")
    with open(json_path, "w") as f:
        json.dump({
            "sign": args.sign,
            "pct": args.pct,
            "variant": args.variant,
            "models": [m for m, *_ in model_data],
            "tasks": all_tasks,
            "avg_matrix": [[None if np.isnan(v) else v for v in row]
                           for row in avg_matrix],
            "count_matrix": count_matrix.tolist(),
        }, f, indent=2)
    print(f"  Saved: {json_path}")

    # Quick summary
    diag_vals = [avg_matrix[i, i] for i in range(n) if not np.isnan(avg_matrix[i, i])]
    off_vals = []
    for i in range(n):
        for j in range(n):
            if i != j and not np.isnan(avg_matrix[i, j]):
                off_vals.append(avg_matrix[i, j])
    print(f"\n  Diagonal (self-ablation):  mean={np.mean(diag_vals):.4f}, n={len(diag_vals)}")
    print(f"  Off-diagonal:              mean={np.mean(off_vals):.4f}, n={len(off_vals)}")
    min_count = count_matrix.min()
    max_count = count_matrix.max()
    print(f"  Cell model-count range: {min_count}–{max_count} (models contributing per cell)")


if __name__ == "__main__":
    main()
