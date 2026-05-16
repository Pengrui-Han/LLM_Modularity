"""3-bar within-vs-across ablation plot.

Per domain D, three bars:
  1. Within:       mean of cells [i, j] with i, j in D, i != j
  2. D -> Others:  mean of cells [i, j] with i in D, j not in D
  3. Others -> D:  mean of cells [i, j] with i not in D, j in D

Runs on either an averaged ablation matrix or a single model's matrix.

Usage:
    python scripts/plot_ablation_within_vs_across_3bar.py --average
    python scripts/plot_ablation_within_vs_across_3bar.py --model Qwen_Qwen2-5-32B-Instruct
"""

import argparse
import csv
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


DOMAINS = ["Lan", "MD", "phys", "ToM"]
DOMAIN_DISPLAY = {"Lan": "Language", "MD": "Formal",
                  "phys": "Physics", "ToM": "Social"}
DOMAIN_COLORS = {"Lan": "#C0392B", "MD": "#2471A3",
                 "phys": "#E67E22", "ToM": "#27AE60"}


def load_matrix(csv_path):
    """Load ablation CSV. Empty / 'nan' cells -> np.nan."""
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        col_labels = header[1:]
        row_labels = []
        rows = []
        for row in reader:
            row_labels.append(row[0])
            vals = []
            for x in row[1:]:
                x = x.strip()
                if x == "" or x.lower() == "nan":
                    vals.append(np.nan)
                else:
                    vals.append(float(x))
            rows.append(vals)
    return np.array(rows), row_labels, col_labels


def domain_of(label):
    return label.split("/")[0]


def compute_bars(matrix, row_labels, col_labels):
    """Returns dict: {domain: {'within': (mean, sem, n),
                              'd_to_others': (...), 'others_to_d': (...)}}"""
    n_rows, n_cols = matrix.shape
    row_dom = [domain_of(l) for l in row_labels]
    col_dom = [domain_of(l) for l in col_labels]

    result = {}
    for D in DOMAINS:
        within_vals = []
        d_to_others_vals = []
        others_to_d_vals = []
        for i in range(n_rows):
            for j in range(n_cols):
                if row_labels[i] == col_labels[j]:
                    continue  # exclude diagonal (task = itself)
                v = matrix[i, j]
                if np.isnan(v):
                    continue
                ri, cj = row_dom[i], col_dom[j]
                if ri == D and cj == D:
                    within_vals.append(v)
                elif ri == D and cj != D:
                    d_to_others_vals.append(v)
                elif ri != D and cj == D:
                    others_to_d_vals.append(v)

        def stats(vals):
            if not vals:
                return (np.nan, np.nan, 0)
            arr = np.array(vals)
            return (arr.mean(), arr.std(ddof=1) / np.sqrt(len(arr)) if len(arr) > 1 else 0.0,
                    len(arr))

        result[D] = {
            "within": stats(within_vals),
            "d_to_others": stats(d_to_others_vals),
            "others_to_d": stats(others_to_d_vals),
        }
    return result


def plot_bars(stats, title, out_path_base):
    # --- PNAS / Nature style ---
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "axes.linewidth": 1.5,
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
        "xtick.major.size": 5,
        "ytick.major.size": 5,
        "hatch.linewidth": 1.0,  # PDF backend drops thin hatches by default
    })

    fig, ax = plt.subplots(figsize=(8, 4.8))

    bar_types = [("within", "Within-Domain", "", 1.0),
                 ("d_to_others", "D \u2192 Others", "///", 0.45),
                 ("others_to_d", "Others \u2192 D", "\\\\\\", 0.45)]

    n_domains = len(DOMAINS)
    n_bars = len(bar_types)
    group_width = 0.75
    bar_width = group_width / n_bars

    x = np.arange(n_domains)

    from matplotlib.colors import to_rgb
    for bi, (key, label, hatch, alpha) in enumerate(bar_types):
        means = [stats[D][key][0] for D in DOMAINS]
        sems = [stats[D][key][1] for D in DOMAINS]
        offset = (bi - (n_bars - 1) / 2) * bar_width
        # Bake alpha into fill so edge/hatch stays solid -> hatch visible in PDF
        fill_colors = [(*to_rgb(DOMAIN_COLORS[D]), alpha) for D in DOMAINS]
        edge_colors = [DOMAIN_COLORS[D] for D in DOMAINS]
        bars = ax.bar(x + offset, means, bar_width,
                      yerr=sems,
                      capsize=3,
                      error_kw={"linewidth": 1.3, "capthick": 1.3},
                      color=fill_colors,
                      edgecolor=edge_colors if hatch else "white",
                      linewidth=0.8 if hatch else 0.5,
                      hatch=hatch)
        # Rasterize hatched bars so PDF viewers render hatches reliably.
        if hatch:
            for b in bars:
                b.set_rasterized(True)

    ax.set_xticks(x)
    ax.set_xticklabels([DOMAIN_DISPLAY[D] for D in DOMAINS], fontsize=13)
    ax.set_xlabel("Domain", fontsize=14, fontweight="bold")
    ax.set_ylabel("\u0394 Accuracy", fontsize=14, fontweight="bold")
    ax.tick_params(axis="both", labelsize=12)

    # Only left + bottom axes
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)

    # No grid (Nature/PNAS clean style)

    # Proxy legend handles: neutral gray, solid (no alpha, no rasterize) -> PDF-safe.
    import matplotlib.patches as mpatches
    legend_gray = "#555555"
    proxy_handles = [
        mpatches.Patch(facecolor=legend_gray, edgecolor=legend_gray,
                       label="Within-Domain"),
        mpatches.Patch(facecolor="white", edgecolor=legend_gray,
                       hatch="///", linewidth=1.0, label="D \u2192 Others"),
        mpatches.Patch(facecolor="white", edgecolor=legend_gray,
                       hatch="\\\\\\", linewidth=1.0, label="Others \u2192 D"),
    ]
    ax.legend(handles=proxy_handles, loc="upper right",
              frameon=False, fontsize=14, handlelength=2.2, handleheight=1.4)

    # Print the n and stats for the record
    print(f"\nStats used in plot:")
    for D in DOMAINS:
        for key, label, _, _ in bar_types:
            m, s, n = stats[D][key]
            print(f"  {DOMAIN_DISPLAY[D]:<16s} {label:<14s}: "
                  f"mean={m:.4f}, SEM={s:.4f}, n={n}")

    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{out_path_base}.{ext}", dpi=300, bbox_inches="tight")
        print(f"  Saved: {out_path_base}.{ext}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--average", action="store_true",
                       help="Use the cross-model averaged matrix.")
    group.add_argument("--model", type=str,
                       help="Model short name, e.g. Qwen_Qwen2-5-32B-Instruct")
    parser.add_argument("--pct", type=str, default="0.1")
    parser.add_argument("--variant", type=str, default="corrupted_accuracy")
    parser.add_argument("--sign", type=str, default="positive")
    parser.add_argument("--results-dir", type=str, default="results")
    args = parser.parse_args()

    filename = f"{args.sign}_{args.pct}pct_{args.variant}.csv"

    if args.average:
        csv_path = os.path.join(args.results_dir, "average", "ablation", filename)
        out_dir = os.path.join(args.results_dir, "average", "figures")
        title = (f"Within-Domain vs Across-Domain Ablation Effect\n"
                 f"(averaged across models, top {args.pct}%)")
        out_tag = "avg"
    else:
        csv_path = os.path.join(args.results_dir, args.model,
                                "ablation_analysis", filename)
        out_dir = os.path.join(args.results_dir, args.model, "figures")
        title = (f"Within-Domain vs Across-Domain Ablation Effect\n"
                 f"{args.model} (top {args.pct}%)")
        out_tag = args.model

    if not os.path.exists(csv_path):
        print(f"Error: CSV not found: {csv_path}")
        sys.exit(1)

    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading: {csv_path}")
    matrix, row_labels, col_labels = load_matrix(csv_path)
    print(f"  Matrix: {matrix.shape}, rows={len(row_labels)}, cols={len(col_labels)}")

    stats = compute_bars(matrix, row_labels, col_labels)

    base_name = f"{args.sign}_{args.pct}pct_{args.variant}_within_vs_across_3bar"
    out_base = os.path.join(out_dir, base_name)
    plot_bars(stats, title, out_base)


if __name__ == "__main__":
    main()
