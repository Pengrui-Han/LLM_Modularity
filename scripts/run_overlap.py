"""
Compute pairwise neuron overlap across all tasks with attribution results.

Usage:
    # Positive top 1%
    python scripts/run_overlap.py --model Qwen/Qwen2.5-32B-Instruct --pct 1 --sign positive --component neurons

    #python scripts/run_overlap.py --model allenai/OLMo-2-0325-32B-Instruct --pct 1 --sign positive --component heads

    # Absolute top 2%
    python scripts/run_overlap.py --model Qwen/Qwen2.5-32B-Instruct --pct 2 --sign absolute

    # Multiple percentages
    python scripts/run_overlap.py --model Qwen/Qwen2.5-32B-Instruct --pct 0.5 1 2 5 --sign positive

    # Overwrite cache
    python scripts/run_overlap.py --model Qwen/Qwen2.5-32B-Instruct --pct 1 --sign positive --overwrite-cache
"""

import argparse
import os
import sys
import json
import torch
import numpy as np
import csv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================
# Tasks to exclude from overlap analysis.
# Add entries as "domain/task_name", e.g. "MD/some_task"
# ============================================================

EXCLUDE_TASKS = {
    # "MD/some_task",
    # "ToM/another_task",
    'ToM/r_agent',
    'ToM/r_social_interactions',
    'ToM/r_social_relations',
    'ToM/mixed_social_relations',
    'ToM/mixed_agent',
    'ToM/mixed_social_interactions',
    'ToM/mixed_social_relations',
    'phys/mixed_spatial',
    'phys/physical_material_behav',
    'phys/physical_obj_motion_force',
    'phys/physical_spatial_relational',
    'phys/material',
    'phys/spatial',
    'ToM/socialqa'

    
}

def get_model_short_name(model_name):
    return model_name.replace("/", "_").replace(".", "-")


def find_all_tasks_with_attribution(results_dir, model_short, component="neurons"):
    """
    Scan results directory and find all tasks that have attribution scores.

    Returns:
        list of (domain, task_name, scores_path) sorted by domain then task
    """
    model_dir = os.path.join(results_dir, model_short)
    if not os.path.exists(model_dir):
        print(f"Error: no results found at {model_dir}")
        sys.exit(1)

    filename = "neuron_attribution.pt" if component == "neurons" else "head_attribution.pt"

    tasks = []
    domains = ["MD", "ToM", "phys", "Lan"]

    for domain in domains:
        domain_dir = os.path.join(model_dir, domain)
        if not os.path.isdir(domain_dir):
            continue
        for task_name in sorted(os.listdir(domain_dir)):
            label = f"{domain}/{task_name}"
            if label in EXCLUDE_TASKS:
                print(f"    [excluded] {label}")
                continue
            scores_path = os.path.join(domain_dir, task_name, filename)
            if os.path.exists(scores_path):
                tasks.append((domain, task_name, scores_path))

    return tasks


def get_cache_path(scores_path, sign, component):
    """Return the cache file path for a given attribution file and sign."""
    task_dir = os.path.dirname(scores_path)
    comp_suffix = "" if component == "neurons" else "_heads"
    return os.path.join(task_dir, f"sorted_indices_{sign}{comp_suffix}.npy")


def get_meta_path(scores_path, component):
    """Return the metadata file path (stores total_units) for a given attribution file."""
    task_dir = os.path.dirname(scores_path)
    comp_suffix = "" if component == "neurons" else "_heads"
    return os.path.join(task_dir, f"attribution_meta{comp_suffix}.json")


def get_sorted_indices(scores_path, sign, component, overwrite_cache=False):
    """
    Load or compute the sorted unit indices for a given attribution file.

    Cache stores:
      - sorted_indices_{sign}.npy: sorted (layer_idx, unit_idx) pairs, shape (N, 2) int32
      - attribution_meta.json: total_units (num_layers * num_units)

    This is pct-agnostic — different pct values just slice [:top_k] from the list.
    On cache hit, attribution.pt is never touched.

    Args:
        scores_path: path to neuron_attribution.pt
        sign: 'positive', 'negative', or 'absolute'
        component: 'neurons' or 'heads'
        overwrite_cache: if True, recompute and overwrite existing cache

    Returns:
        (sorted_indices, total_units): np.ndarray of shape (N, 2), int
    """
    cache_path = get_cache_path(scores_path, sign, component)
    meta_path = get_meta_path(scores_path, component)

    if not overwrite_cache and os.path.exists(cache_path) and os.path.exists(meta_path):
        sorted_indices = np.load(cache_path)
        with open(meta_path, "r") as f:
            total_units = json.load(f)["total_units"]
        return sorted_indices, total_units

    # Load attribution and compute sorted indices
    attribution = torch.load(scores_path, map_location="cpu").numpy()
    num_layers, num_units = attribution.shape
    total_units = num_layers * num_units

    # Save metadata (shared across all signs for this component)
    with open(meta_path, "w") as f:
        json.dump({"total_units": total_units, "num_layers": num_layers, "num_units": num_units}, f)

    layer_idx, unit_idx = np.meshgrid(
        np.arange(num_layers), np.arange(num_units), indexing="ij"
    )
    layer_flat = layer_idx.ravel()
    unit_flat = unit_idx.ravel()
    score_flat = attribution.ravel()

    if sign == "positive":
        mask = score_flat > 0
        order = np.argsort(-score_flat[mask])
        layer_sel = layer_flat[mask][order]
        unit_sel = unit_flat[mask][order]
    elif sign == "negative":
        mask = score_flat < 0
        order = np.argsort(score_flat[mask])
        layer_sel = layer_flat[mask][order]
        unit_sel = unit_flat[mask][order]
    elif sign == "absolute":
        order = np.argsort(-np.abs(score_flat))
        layer_sel = layer_flat[order]
        unit_sel = unit_flat[order]
    else:
        raise ValueError(f"sign must be 'positive', 'negative', or 'absolute'")

    sorted_indices = np.stack([layer_sel, unit_sel], axis=1).astype(np.int32)

    np.save(cache_path, sorted_indices)
    print(f"    [cache saved] {os.path.basename(cache_path)}")

    return sorted_indices, total_units


def get_top_neurons(sorted_indices, percentage, total_units):
    """
    Get top neurons as a set of (layer, neuron_idx) tuples from pre-sorted indices.

    Args:
        sorted_indices: np.ndarray of shape (N, 2)
        percentage: float
        total_units: int — num_layers * num_units (for computing top_k)

    Returns:
        set of (layer_idx, neuron_idx), top_k
    """
    top_k = max(1, int(total_units * percentage / 100.0))
    selected = sorted_indices[:top_k]
    return set(map(tuple, selected)), top_k


def compute_overlap_matrix(task_neuron_sets):
    """
    Compute pairwise overlap ratio matrix.

    Args:
        task_neuron_sets: list of (label, neuron_set, top_k)

    Returns:
        np.ndarray [N, N], list of labels
    """
    n = len(task_neuron_sets)
    matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            set_i = task_neuron_sets[i][1]
            set_j = task_neuron_sets[j][1]
            k_i = task_neuron_sets[i][2]

            intersection = len(set_i & set_j)
            matrix[i, j] = intersection / k_i if k_i > 0 else 0.0

    labels = [t[0] for t in task_neuron_sets]
    return matrix, labels


def find_domain_boundaries(labels):
    """
    Find boundaries between domains for drawing separator lines.

    Args:
        labels: list of "domain/task" strings

    Returns:
        list of boundary indices where a new domain starts
    """
    boundaries = []
    prev_domain = None
    for i, label in enumerate(labels):
        domain = label.split("/")[0]
        if domain != prev_domain:
            if prev_domain is not None:
                boundaries.append(i)
            prev_domain = domain
    return boundaries


def run_overlap(args, pct):
    """Run overlap analysis for a single percentage."""
    model_short = get_model_short_name(args.model)

    # Find all tasks
    tasks = find_all_tasks_with_attribution(args.results_dir, model_short, args.component)

    if len(tasks) < 2:
        print(f"  Need at least 2 tasks with attribution results, found {len(tasks)}")
        return

    print(f"\n  Found {len(tasks)} tasks with attribution scores:")
    for domain, task_name, _ in tasks:
        print(f"    {domain}/{task_name}")

    # Load sorted indices (from cache or computed fresh) and get top neurons
    task_neuron_sets = []
    for domain, task_name, scores_path in tasks:
        sorted_indices, total_units = get_sorted_indices(
            scores_path, args.sign, args.component, overwrite_cache=args.overwrite_cache
        )

        neuron_set, top_k = get_top_neurons(sorted_indices, pct, total_units)
        label = f"{domain}/{task_name}"
        task_neuron_sets.append((label, neuron_set, top_k))
        print(f"    {label}: {top_k} neurons selected (top {pct}%)")

    # Compute overlap matrix
    matrix, labels = compute_overlap_matrix(task_neuron_sets)

    # --- Save CSV ---
    output_dir = os.path.join(args.results_dir, model_short, "overlap")
    os.makedirs(output_dir, exist_ok=True)

    comp_suffix = "" if args.component == "neurons" else f"_{args.component}"

    csv_path = os.path.join(output_dir, f"{args.sign}_{pct}pct{comp_suffix}_overlap.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([""] + labels)
        for i, label in enumerate(labels):
            row = [label] + [f"{matrix[i, j]:.4f}" for j in range(len(labels))]
            writer.writerow(row)
    print(f"  ✓ Saved: {csv_path}")

    # --- Save JSON with metadata ---
    json_path = os.path.join(output_dir, f"{args.sign}_{pct}pct{comp_suffix}_overlap.json")
    with open(json_path, "w") as f:
        json.dump({
            "model": args.model,
            "sign": args.sign,
            "percentage": pct,
            "component": args.component,
            "tasks": labels,
            "overlap_matrix": matrix.tolist(),
        }, f, indent=2)
    print(f"  ✓ Saved: {json_path}")

    # --- Heatmap ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    short_labels = []
    for label in labels:
        domain, task = label.split("/", 1)
        if len(task) > 20:
            task = task[:17] + "..."
        short_labels.append(f"{domain}\n{task}")

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.2), max(6, len(labels) * 1.0)))

    im = ax.imshow(matrix, cmap="YlOrRd", vmin=0, vmax=1, aspect="equal")

    boundaries = find_domain_boundaries(labels)
    for b in boundaries:
        ax.axhline(y=b - 0.5, color="black", linewidth=2)
        ax.axvline(x=b - 0.5, color="black", linewidth=2)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(short_labels, fontsize=8)

    for i in range(len(labels)):
        for j in range(len(labels)):
            val = matrix[i, j]
            color = "white" if val > 0.5 else "black"
            ax.text(j, i, f"{val:.1%}", ha="center", va="center",
                    fontsize=7, color=color)

    plt.colorbar(im, ax=ax, label="Overlap Ratio", shrink=0.8)
    comp_label = "Head" if args.component == "heads" else "Neuron"
    ax.set_title(f"{comp_label} Overlap ({args.sign} top {pct}%)\n{args.model}", fontsize=11)
    ax.set_xlabel("Task B (target)")
    ax.set_ylabel("Task A (source)")

    plt.tight_layout()
    png_path = os.path.join(output_dir, f"{args.sign}_{pct}pct{comp_suffix}_overlap.png")
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: {png_path}")

    # --- Print summary ---
    print(f"\n  OVERLAP MATRIX ({args.sign} top {pct}%):")
    print(f"  {'':>25s}", end="")
    for label in labels:
        short = label.split("/")[1][:10]
        print(f"  {short:>10s}", end="")
    print()

    for i, label_i in enumerate(labels):
        short_i = label_i.split("/")[1][:25]
        print(f"  {short_i:>25s}", end="")
        for j in range(len(labels)):
            val = matrix[i, j]
            print(f"  {val:>9.1%}", end="")
        print()

    same_domain_vals = []
    cross_domain_vals = []
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i == j:
                continue
            domain_i = labels[i].split("/")[0]
            domain_j = labels[j].split("/")[0]
            if domain_i == domain_j:
                same_domain_vals.append(matrix[i, j])
            else:
                cross_domain_vals.append(matrix[i, j])

    if same_domain_vals:
        print(f"\n  Same-domain overlap:  mean={np.mean(same_domain_vals):.1%}, "
              f"min={np.min(same_domain_vals):.1%}, max={np.max(same_domain_vals):.1%}")
    if cross_domain_vals:
        print(f"  Cross-domain overlap: mean={np.mean(cross_domain_vals):.1%}, "
              f"min={np.min(cross_domain_vals):.1%}, max={np.max(cross_domain_vals):.1%}")


def main():
    parser = argparse.ArgumentParser(description="Compute pairwise neuron overlap")
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--pct", type=float, nargs="+", default=[1.0],
                        help="Percentage(s) for top-k selection")
    parser.add_argument("--sign", default="positive",
                        choices=["positive", "negative", "absolute"])
    parser.add_argument("--component", default="neurons",
                        choices=["neurons", "heads"])
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--overwrite-cache", action="store_true",
                        help="Recompute and overwrite sorted index cache")
    args = parser.parse_args()

    print("=" * 80)
    print("PAIRWISE NEURON OVERLAP ANALYSIS")
    print("=" * 80)
    print(f"  Model: {args.model}")
    print(f"  Sign: {args.sign}")
    print(f"  Percentages: {args.pct}")
    print(f"  Component: {args.component}")
    print(f"  Overwrite cache: {args.overwrite_cache}")

    for pct in args.pct:
        print(f"\n{'='*80}")
        print(f"  PERCENTAGE: {pct}%")
        print(f"{'='*80}")
        run_overlap(args, pct)

    print(f"\n{'='*80}")
    print("OVERLAP ANALYSIS COMPLETE!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()