"""Cross-model Kendall's tau on ablation matrices.

For each pair of models, restrict to the common task set, take the off-diagonal
ablation values (raw, non-symmetrized — ablation is directional), and compute
Kendall's tau rank correlation.

15 model pairs. Reports per-pair tau + p, and overall mean ± std.
"""

import csv
import os
import sys

import numpy as np
from scipy.stats import kendalltau


MODELS = [
    ("Qwen_Qwen2-5-32B-Instruct", "Qwen2.5-32B"),
    ("allenai_OLMo-2-0325-32B-Instruct", "OLMo-32B"),
    ("mistralai_Mistral-Small-24B-Instruct-2501", "Mistral-24B"),
    ("Qwen_Qwen2-5-72B-Instruct", "Qwen2.5-72B"),
    ("meta-llama_Meta-Llama-3-1-70B-Instruct", "Llama-70B"),
    ("mistralai_Mistral-Large-Instruct-2407", "Mistral-123B"),
]
CSV_REL = "ablation_analysis/positive_0.1pct_corrupted_accuracy.csv"


def load_matrix(csv_path):
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        col_labels = header[1:]
        row_labels = []
        rows = []
        for r in reader:
            row_labels.append(r[0])
            vals = []
            for x in r[1:]:
                x = x.strip()
                vals.append(np.nan if x == "" or x.lower() == "nan" else float(x))
            rows.append(vals)
    return np.array(rows), row_labels, col_labels


def main():
    matrices = {}
    for short, name in MODELS:
        path = os.path.join("results", short, CSV_REL)
        if not os.path.exists(path):
            print(f"  Missing: {path}")
            continue
        M, rows, cols = load_matrix(path)
        if rows != cols:
            print(f"  WARN: rows != cols for {name}, using row labels")
        matrices[name] = (rows, M)
        print(f"  {name}: {M.shape}, {len(rows)} tasks")

    if len(matrices) < 2:
        print("Not enough models loaded.")
        sys.exit(1)

    # Common tasks across all loaded models
    common = sorted(set.intersection(*[set(rows) for rows, _ in matrices.values()]))
    print(f"\nCommon tasks across {len(matrices)} models: {len(common)}")

    # Build per-model off-diagonal vector restricted to common tasks
    vecs = {}
    for name, (rows, M) in matrices.items():
        idx = [rows.index(t) for t in common]
        sub = M[np.ix_(idx, idx)]
        # Off-diagonal vector (all i != j entries, preserving direction)
        n = len(common)
        mask = ~np.eye(n, dtype=bool)
        vecs[name] = sub[mask]

    names = list(matrices.keys())
    print(f"\nPairwise Kendall's tau (off-diag, directional, raw values):")
    taus = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            v1 = vecs[names[i]]
            v2 = vecs[names[j]]
            # Drop pairs where either value is NaN (shouldn't happen in per-model CSVs but safe)
            ok = ~(np.isnan(v1) | np.isnan(v2))
            tau, p = kendalltau(v1[ok], v2[ok])
            taus.append(tau)
            print(f"  {names[i]:>14s} vs {names[j]:>14s}: tau={tau:.4f}, p={p:.2e}, n={ok.sum()}")

    taus = np.array(taus)
    print(f"\nMean tau: {taus.mean():.4f} +/- {taus.std():.4f}")
    print(f"Range: {taus.min():.4f} - {taus.max():.4f}")
    print(f"Number of pairs: {len(taus)}")


if __name__ == "__main__":
    main()
