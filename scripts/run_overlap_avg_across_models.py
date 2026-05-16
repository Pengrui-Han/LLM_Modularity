"""
Average doubly-stochastic overlap matrices across models.

For each (task_i, task_j) pair, averages over all models that have both tasks.
Automatically discovers model directories under results/.

Usage:
    python scripts/run_overlap_avg_across_models.py --pct 1.0 --sign positive --component neurons
    python scripts/run_overlap_avg_across_models.py --pct 1.0 --sign positive --component heads
    python scripts/run_overlap_avg_across_models.py --pct 0.5 1.0 2.0 --sign positive --component neurons
"""

import argparse
import os
import sys
import json
import numpy as np
import csv
import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_overlap_csv(csv_path):
    """Load overlap CSV into matrix + labels."""
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        labels = header[1:]
        matrix = []
        for row in reader:
            matrix.append([float(x) for x in row[1:]])
    return np.array(matrix), labels


def find_ds_csvs(results_dir, sign, pct, component, raw=False):
    """
    Auto-discover overlap CSVs across model directories.

    Looks for: results/<model_short>/overlap/<sign>_<pct>pct[_<comp>]_overlap[_doubly_stochastic].csv

    Returns:
        list of (model_name, csv_path)
    """
    comp_suffix = "" if component == "neurons" else f"_{component}"
    # Format pct: 1.0 -> "1.0", 0.5 -> "0.5"
    pct_str = f"{pct}".rstrip("0").rstrip(".") if pct == int(pct) else f"{pct}"
    # Actually keep consistent with how run_overlap.py formats it (uses Python's default float str)
    pct_str = str(pct)

    if raw:
        filename = f"{sign}_{pct_str}pct{comp_suffix}_overlap.csv"
    else:
        filename = f"{sign}_{pct_str}pct{comp_suffix}_overlap_doubly_stochastic.csv"

    found = []
    if not os.path.isdir(results_dir):
        print(f"Error: results directory not found: {results_dir}")
        sys.exit(1)

    allowed_models = {
        "Qwen_Qwen2-5-32B-Instruct",
        "Qwen_Qwen2-5-72B-Instruct",
        "allenai_OLMo-2-0325-32B-Instruct",
        "meta-llama_Meta-Llama-3-1-70B-Instruct",
        "mistralai_Mistral-Large-Instruct-2407",
        "mistralai_Mistral-Small-24B-Instruct-2501",
    }
    for model_dir_name in sorted(os.listdir(results_dir)):
        if model_dir_name not in allowed_models:
            continue
        csv_path = os.path.join(results_dir, model_dir_name, "overlap", filename)
        if os.path.exists(csv_path):
            found.append((model_dir_name, csv_path))

    return found


def find_domain_boundaries(labels):
    """Find indices where domain changes."""
    boundaries = []
    prev_domain = None
    for i, label in enumerate(labels):
        domain = label.split("/")[0]
        if domain != prev_domain:
            if prev_domain is not None:
                boundaries.append(i)
            prev_domain = domain
    return boundaries


def domain_sort_key(label):
    """Sort key: domain order first, then task name."""
    domain_order = {"MD": 0, "ToM": 1, "phys": 2, "Lan": 3}
    domain = label.split("/")[0]
    task = label.split("/")[1] if "/" in label else ""
    return (domain_order.get(domain, 99), task)


def plot_heatmap(matrix, labels, title, output_path, count_matrix=None):
    """Plot heatmap with domain separators and optional count annotations."""
    short_labels = []
    for label in labels:
        domain, task = label.split("/", 1)
        if len(task) > 20:
            task = task[:17] + "..."
        short_labels.append(f"{domain}\n{task}")

    n = len(labels)
    fig, ax = plt.subplots(figsize=(max(8, n * 1.2), max(6, n * 1.0)))

    im = ax.imshow(matrix, cmap="YlOrRd", aspect="equal")

    # Domain separator lines
    boundaries = find_domain_boundaries(labels)
    for b in boundaries:
        ax.axhline(y=b - 0.5, color="black", linewidth=2)
        ax.axvline(x=b - 0.5, color="black", linewidth=2)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(short_labels, fontsize=8)

    # Text annotations
    vmax = matrix.max() if matrix.max() > 0 else 1.0
    for i in range(n):
        for j in range(n):
            val = matrix[i, j]
            color = "white" if val > 0.5 * vmax else "black"
            if count_matrix is not None and i != j:
                cnt = int(count_matrix[i, j])
                text = f"{val:.3f}\n({cnt})"
            else:
                text = f"{val:.3f}"
            ax.text(j, i, text, ha="center", va="center",
                    fontsize=6, color=color)

    plt.colorbar(im, ax=ax, label="Avg Doubly Stochastic Normalized", shrink=0.8)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Task B")
    ax.set_ylabel("Task A")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: {output_path}")


def run_avg(args, pct):
    """Run cross-model averaging for a single percentage."""

    # Step 1: Find all CSVs
    found = find_ds_csvs(args.results_dir, args.sign, pct, args.component, raw=args.raw)

    if len(found) == 0:
        print(f"  No doubly stochastic CSVs found for pct={pct}")
        return
    
    print(f"\n  Found {len(found)} models:")
    for model_name, csv_path in found:
        print(f"    {model_name}: {csv_path}")

    # Step 2: Load all matrices
    model_data = []  # list of (model_name, matrix, labels)
    for model_name, csv_path in found:
        matrix, labels = load_overlap_csv(csv_path)
        model_data.append((model_name, matrix, labels))
        print(f"    {model_name}: {len(labels)} tasks")

    # Step 3: Collect task union, sorted by domain order
    all_tasks = set()
    for _, _, labels in model_data:
        all_tasks.update(labels)
    all_tasks = sorted(all_tasks, key=domain_sort_key)
    n = len(all_tasks)
    task_to_idx = {t: i for i, t in enumerate(all_tasks)}

    print(f"\n  Task union: {n} tasks")
    for t in all_tasks:
        # Show which models have this task
        models_with = [m for m, _, labels in model_data if t in labels]
        print(f"    {t}: {len(models_with)} models")

    # Step 4: Accumulate sums and counts
    sum_matrix = np.zeros((n, n))
    count_matrix = np.zeros((n, n))

    for model_name, matrix, labels in model_data:
        label_to_local = {l: i for i, l in enumerate(labels)}
        for ti in labels:
            for tj in labels:
                gi = task_to_idx[ti]
                gj = task_to_idx[tj]
                li = label_to_local[ti]
                lj = label_to_local[tj]
                sum_matrix[gi, gj] += matrix[li, lj]
                count_matrix[gi, gj] += 1

    # Step 5: Compute average (avoid division by zero)
    avg_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if count_matrix[i, j] > 0:
                avg_matrix[i, j] = sum_matrix[i, j] / count_matrix[i, j]

    # Step 6: Output
    comp_suffix = "" if args.component == "neurons" else f"_{args.component}"
    pct_str = str(pct)

    output_dir = os.path.join(args.results_dir, "average", "overlap")
    os.makedirs(output_dir, exist_ok=True)

    if args.raw:
        base_name = f"{args.sign}_{pct_str}pct{comp_suffix}_overlap"
    else:
        base_name = f"{args.sign}_{pct_str}pct{comp_suffix}_overlap_doubly_stochastic"

    # CSV
    csv_path = os.path.join(output_dir, f"{base_name}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([""] + all_tasks)
        for i, task in enumerate(all_tasks):
            row = [task] + [f"{avg_matrix[i, j]:.6f}" for j in range(n)]
            writer.writerow(row)
    print(f"  ✓ Saved: {csv_path}")

    # JSON with metadata
    json_path = os.path.join(output_dir, f"{base_name}.json")
    with open(json_path, "w") as f:
        json.dump({
            "sign": args.sign,
            "percentage": pct,
            "component": args.component,
            "models": [m for m, _, _ in model_data],
            "tasks": all_tasks,
            "avg_overlap_matrix": avg_matrix.tolist(),
            "count_matrix": count_matrix.astype(int).tolist(),
        }, f, indent=2)
    print(f"  ✓ Saved: {json_path}")

    # Heatmap
    model_list_str = ", ".join(m for m, _, _ in model_data)
    title = f"Avg Doubly Stochastic Overlap ({args.sign} top {pct}%)\n{len(model_data)} models: {model_list_str}"
    # Truncate title if too long
    if len(title) > 150:
        title = f"Avg Doubly Stochastic Overlap ({args.sign} top {pct}%)\n{len(model_data)} models"

    png_path = os.path.join(output_dir, f"{base_name}.png")
    plot_heatmap(avg_matrix, all_tasks, title, png_path, count_matrix=count_matrix)

    # Print summary
    print(f"\n  AVERAGE OVERLAP MATRIX ({args.sign} top {pct}%, {len(model_data)} models):")

    same_domain_vals = []
    cross_domain_vals = []
    for i in range(n):
        for j in range(n):
            if i == j or count_matrix[i, j] == 0:
                continue
            domain_i = all_tasks[i].split("/")[0]
            domain_j = all_tasks[j].split("/")[0]
            if domain_i == domain_j:
                same_domain_vals.append(avg_matrix[i, j])
            else:
                cross_domain_vals.append(avg_matrix[i, j])

    if same_domain_vals:
        print(f"  Same-domain overlap:  mean={np.mean(same_domain_vals):.4f}, "
              f"min={np.min(same_domain_vals):.4f}, max={np.max(same_domain_vals):.4f}")
    if cross_domain_vals:
        print(f"  Cross-domain overlap: mean={np.mean(cross_domain_vals):.4f}, "
              f"min={np.min(cross_domain_vals):.4f}, max={np.max(cross_domain_vals):.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Average doubly-stochastic overlap matrices across models"
    )
    parser.add_argument("--pct", type=float, nargs="+", default=[1.0],
                        help="Percentage(s) to process")
    parser.add_argument("--sign", default="positive",
                        choices=["positive", "negative", "absolute"])
    parser.add_argument("--component", default="neurons",
                        choices=["neurons", "heads"])
    parser.add_argument("--raw", action="store_true",
                        help="Use raw overlap CSV instead of doubly stochastic")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    print("=" * 80)
    print("CROSS-MODEL AVERAGE OVERLAP ANALYSIS")
    print("=" * 80)
    print(f"  Sign: {args.sign}")
    print(f"  Percentages: {args.pct}")
    print(f"  Component: {args.component}")
    print(f"  Results dir: {args.results_dir}")

    for pct in args.pct:
        print(f"\n{'=' * 80}")
        print(f"  PERCENTAGE: {pct}%")
        print(f"{'=' * 80}")
        run_avg(args, pct)

    print(f"\n{'=' * 80}")
    print("CROSS-MODEL AVERAGING COMPLETE!")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()