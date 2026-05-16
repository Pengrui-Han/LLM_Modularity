"""
Ablation analysis: generate heatmaps of ablation effects across all source-target task pairs.

Usage:
    # All signs, 1% ablation
    python scripts/run_ablation_analysis.py --model Qwen/Qwen2.5-32B-Instruct --pct 1

    # Specific sign
    python scripts/run_ablation_analysis.py --model mistralai/Mistral-Small-24B-Instruct-2501 --pct 1 --sign positive --component heads

    # Multiple percentages (one heatmap per sign per pct)
    python scripts/run_ablation_analysis.py --model Qwen/Qwen2.5-32B-Instruct --pct 1 5
"""

import argparse
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================
# Tasks to exclude from ablation analysis (both as source and target).
# Kept in sync with EXCLUDE_TASKS in run_overlap.py.
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


def get_model_short_name(model_name):
    return model_name.replace("/", "_").replace(".", "-")


def collect_ablation_results(results_base, model_short, component, sign, pct, ablation_type="zero"):
    """
    Scan ablation results directory and collect all source-target pairs.

    Returns:
        dict: {(source_domain, source_task, target_domain, target_task): result_dict}
    """
    ablation_root = os.path.join(results_base, model_short, "ablation")
    results = {}

    if not os.path.exists(ablation_root):
        print(f"  No ablation results found at {ablation_root}")
        return results

    # Walk: ablation/{source_domain}/{source_task}/{component}_ablated_on_{target_domain}_{target_task}/
    for source_domain in sorted(os.listdir(ablation_root)):
        source_domain_path = os.path.join(ablation_root, source_domain)
        if not os.path.isdir(source_domain_path):
            continue

        for source_task in sorted(os.listdir(source_domain_path)):
            source_task_path = os.path.join(source_domain_path, source_task)
            if not os.path.isdir(source_task_path):
                continue

            for ablation_dir_name in sorted(os.listdir(source_task_path)):
                # Parse: {component}_ablated_on_{target_domain}_{target_task}
                prefix = f"{component}_ablated_on_"
                if not ablation_dir_name.startswith(prefix):
                    continue

                target_part = ablation_dir_name[len(prefix):]
                # target_part = "MD_add_sub_2op_symbolic" etc.
                # We need to split domain from task: first part before _ is domain
                # But domain names like "MD", "ToM", "phys", "Lan" are known
                target_domain = None
                target_task = None
                for domain in ["MD", "ToM", "phys", "Lan"]:
                    if target_part.startswith(domain + "_"):
                        target_domain = domain
                        target_task = target_part[len(domain) + 1:]
                        break

                if target_domain is None:
                    continue

                # Skip if either source or target is in the exclude list
                if f"{source_domain}/{source_task}" in EXCLUDE_TASKS:
                    continue
                if f"{target_domain}/{target_task}" in EXCLUDE_TASKS:
                    continue

                # Look for result file
                if ablation_type == "zero":
                    filename = f"{sign}_{pct}pct.json"
                else:
                    filename = f"{sign}_{pct}pct_{ablation_type}.json"
                result_file = os.path.join(
                    source_task_path, ablation_dir_name, filename
                )
                if os.path.exists(result_file):
                    with open(result_file, "r") as f:
                        data = json.load(f)
                    results[(source_domain, source_task, target_domain, target_task)] = data

    return results


def build_heatmap(results, metric="accuracy", mode="vs_random"):
    """
    Build a heatmap matrix from collected results.

    Args:
        results: dict from collect_ablation_results
        metric: 'accuracy' or 'score'
        mode: 'vs_random' (target - random), 'raw' (no_ablation - target, i.e. accuracy drop),
              or 'raw_accuracy' (just target accuracy)

    Returns:
        matrix: np.ndarray [num_sources, num_targets]
        source_labels: list of "domain/task" strings
        target_labels: list of "domain/task" strings
    """
    # Collect all unique source and target tasks
    source_set = set()
    target_set = set()
    for (sd, st, td, tt) in results.keys():
        source_set.add((sd, st))
        target_set.add((td, tt))

    # Sort: group by domain, then alphabetically
    source_labels = sorted(source_set, key=lambda x: (x[0], x[1]))
    target_labels = sorted(target_set, key=lambda x: (x[0], x[1]))

    source_idx = {k: i for i, k in enumerate(source_labels)}
    target_idx = {k: i for i, k in enumerate(target_labels)}

    matrix = np.full((len(source_labels), len(target_labels)), np.nan)

    for (sd, st, td, tt), data in results.items():
        si = source_idx[(sd, st)]
        ti = target_idx[(td, tt)]

        if mode == "raw":
            # Accuracy drop: 1.0 - target (both_correct subset: no_ablation = 1.0 by definition)
            if metric == "accuracy":
                matrix[si, ti] = 1.0 - data["target"]["accuracy"]
            else:
                matrix[si, ti] = 1.0 - data["target"]["score"]
        elif mode == "raw_accuracy":
            # Just target value after ablation
            if metric == "accuracy":
                matrix[si, ti] = data["target"]["accuracy"]
            else:
                matrix[si, ti] = data["target"]["score"]
        else:
            # vs_random: target - random
            if metric == "accuracy":
                target_val = data["target"]["accuracy"]
                random_val = data["truly_random"]["mean_accuracy"]
            else:
                target_val = data["target"]["score"]
                random_val = data["truly_random"]["mean_score"]
            if random_val is not None:
                matrix[si, ti] = target_val - random_val
            else:
                matrix[si, ti] = np.nan

    # Format labels as "domain/task"
    source_strs = [f"{d}/{t}" for d, t in source_labels]
    target_strs = [f"{d}/{t}" for d, t in target_labels]

    return matrix, source_strs, target_strs


def plot_heatmap(matrix, source_labels, target_labels, title, output_path, use_log=False):
    """Plot and save a heatmap."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(max(10, len(target_labels) * 1.2),
                                     max(8, len(source_labels) * 0.8)))

    if use_log:
        from matplotlib.colors import SymLogNorm
        vmax = max(abs(np.nanmin(matrix)), abs(np.nanmax(matrix)))
        if vmax == 0:
            vmax = 1.0
        im = ax.imshow(matrix, cmap="RdBu_r", aspect="auto",
                       norm=SymLogNorm(linthresh=0.01, vmin=-vmax, vmax=vmax))
    else:
        # Use diverging colormap centered at 0
        vmax = max(abs(np.nanmin(matrix)), abs(np.nanmax(matrix)))
        if vmax == 0:
            vmax = 1.0
        im = ax.imshow(matrix, cmap="RdBu_r", aspect="auto",
                       vmin=-vmax, vmax=vmax)

    # Labels
    ax.set_xticks(range(len(target_labels)))
    ax.set_xticklabels(target_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(source_labels)))
    ax.set_yticklabels(source_labels, fontsize=8)

    ax.set_xlabel("Target Task")
    ax.set_ylabel("Source Task")
    ax.set_title(title)

    # Annotate cells
    for i in range(len(source_labels)):
        for j in range(len(target_labels)):
            val = matrix[i, j]
            if not np.isnan(val):
                color = "white" if abs(val) > vmax * 0.6 else "black"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                       fontsize=7, color=color)

    # Add domain separator lines
    def add_separators(labels, axis):
        domains = [l.split("/")[0] for l in labels]
        for i in range(1, len(domains)):
            if domains[i] != domains[i-1]:
                if axis == "x":
                    ax.axvline(x=i - 0.5, color="black", linewidth=1.5)
                else:
                    ax.axhline(y=i - 0.5, color="black", linewidth=1.5)

    add_separators(target_labels, "x")
    add_separators(source_labels, "y")

    plt.colorbar(im, ax=ax, label="Target acc - Truly random acc", shrink=0.8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    ✓ Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Ablation analysis heatmaps")
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--pct", type=float, nargs="+", default=[1.0],
                        help="Percentage(s) to analyze")
    parser.add_argument("--sign", nargs="+", default=["positive", "negative", "absolute"],
                        choices=["positive", "negative", "absolute"],
                        help="Sign(s) to analyze")
    parser.add_argument("--component", default="neurons", choices=["neurons", "heads"])
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--mode", default="vs_random", choices=["vs_random", "raw", "raw_accuracy"],
                        help="'vs_random': target - random; 'raw': accuracy drop (no_ablation - target); "
                             "'raw_accuracy': just target accuracy after ablation")
    parser.add_argument("--ablation-type", default="zero", choices=["zero", "corrupted"],
                        help="Which ablation results to analyze")
    parser.add_argument("--log", action="store_true",
                        help="Use log color scale for heatmap")
    args = parser.parse_args()

    model_short = get_model_short_name(args.model)

    print("=" * 80)
    print("ABLATION ANALYSIS")
    print(f"  Model: {args.model}")
    print(f"  Signs: {args.sign}")
    print(f"  Percentages: {args.pct}")
    print("=" * 80)

    output_dir = os.path.join(args.results_dir, model_short, "ablation_analysis")
    os.makedirs(output_dir, exist_ok=True)

    for pct in args.pct:
        for sign in args.sign:
            print(f"\n  --- {sign} {pct}% ---")

            results = collect_ablation_results(
                args.results_dir, model_short, args.component, sign, pct,
                ablation_type=args.ablation_type
            )

            if not results:
                print(f"    No results found for {sign} {pct}%")
                continue

            print(f"    Found {len(results)} source-target pairs")

            # Build and save heatmap
            matrix, source_labels, target_labels = build_heatmap(results, metric="accuracy", mode=args.mode)

            # Component and ablation type suffix for filenames
            comp_suffix = "" if args.component == "neurons" else f"_{args.component}"
            abl_suffix = "" if args.ablation_type == "zero" else f"_{args.ablation_type}"

            # Save CSV
            csv_path = os.path.join(output_dir, f"{sign}_{pct}pct{comp_suffix}{abl_suffix}_accuracy.csv")
            with open(csv_path, "w") as f:
                f.write("source\\target," + ",".join(target_labels) + "\n")
                for i, sl in enumerate(source_labels):
                    vals = ",".join(
                        f"{matrix[i,j]:.4f}" if not np.isnan(matrix[i,j]) else ""
                        for j in range(len(target_labels))
                    )
                    f.write(f"{sl},{vals}\n")
            print(f"    ✓ Saved: {csv_path}")

            # Plot
            if args.mode == "raw":
                title = f"Ablation Effect: {sign} {pct}% {args.component}\n(accuracy drop, positive = ablation hurts)"
            elif args.mode == "raw_accuracy":
                title = f"Ablation Effect: {sign} {pct}% {args.component}\n(accuracy after ablation)"
            else:
                title = f"Ablation Effect: {sign} {pct}% {args.component}\n(target_acc - random_acc, negative = important)"
            log_suffix = "_log" if args.log else ""
            plot_path = os.path.join(output_dir, f"{sign}_{pct}pct{comp_suffix}{abl_suffix}_accuracy{log_suffix}.png")
            plot_heatmap(matrix, source_labels, target_labels, title, plot_path, use_log=args.log)

            # Print summary statistics
            valid = matrix[~np.isnan(matrix)]
            if len(valid) > 0:
                # Diagonal = same-domain (where source == target)
                diag_vals = []
                off_diag_vals = []
                for i in range(len(source_labels)):
                    for j in range(len(target_labels)):
                        if np.isnan(matrix[i, j]):
                            continue
                        s_domain = source_labels[i].split("/")[0]
                        t_domain = target_labels[j].split("/")[0]
                        if source_labels[i] == target_labels[j]:
                            diag_vals.append(matrix[i, j])
                        elif s_domain == t_domain:
                            # Same domain, different task
                            pass
                        off_diag_vals.append(matrix[i, j])

                print(f"\n    Summary:")
                print(f"      All pairs:       mean={np.mean(valid):.4f}, min={np.min(valid):.4f}, max={np.max(valid):.4f}")
                if diag_vals:
                    print(f"      Same-task (diag): mean={np.mean(diag_vals):.4f}")
                if off_diag_vals:
                    print(f"      Cross-task:       mean={np.mean(off_diag_vals):.4f}")

    print(f"\n{'='*80}")
    print(f"Analysis complete! Results in: {output_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()