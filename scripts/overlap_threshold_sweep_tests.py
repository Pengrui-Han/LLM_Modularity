"""Robustness-to-threshold sweep for the three overlap tests.

For each top-X% threshold in {0.05, 0.1, 1.0, 5.0}, recompute:
  Test 1  per-model within-vs-cross-domain permutation test
          (mirrors per_model_modularity_summary.py, overlap side only)
  Test 2  hierarchical clustering on the model-averaged overlap matrix
          + Adjusted Rand Index against 4 a priori domains
          (mirrors test_ablation_ari.py)
  Test 3  pairwise Kendall's tau across the 6 per-model overlap matrices
          (mirrors test_ablation_cross_model_kendall.py)

Inputs (read-only):
  results/<model>/overlap/positive_<pct>pct_overlap.csv      x 6 models
  results/average/overlap/positive_<pct>pct_overlap.csv

Outputs (NEW files, do not overwrite anything existing):
  docs/threshold_sweep_summary.csv
  docs/threshold_sweep_summary.md
"""

import csv
import os

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import kendalltau
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
    v_measure_score,
)


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODELS = [
    ("Mistral-Small-24B",  "mistralai_Mistral-Small-24B-Instruct-2501",  24),
    ("Qwen2.5-32B",        "Qwen_Qwen2-5-32B-Instruct",                  32),
    ("OLMo-2-32B",         "allenai_OLMo-2-0325-32B-Instruct",           32),
    ("Llama-3.1-70B",      "meta-llama_Meta-Llama-3-1-70B-Instruct",     70),
    ("Qwen2.5-72B",        "Qwen_Qwen2-5-72B-Instruct",                  72),
    ("Mistral-Large-123B", "mistralai_Mistral-Large-Instruct-2407",     123),
]

PCTS    = [0.05, 0.1, 1.0, 5.0]
N_PERM  = 10000
SEED    = 42
ARI_K   = 5  # kNN for graph used in test_ablation_ari.py


def pct_token(p):
    """Match the on-disk filename convention: 0.05 -> '0.05', 0.1 -> '0.1',
    1.0 -> '1.0', 5.0 -> '5.0'."""
    return f"{p}" if p < 1 else f"{p:.1f}"


# ---------------------------------------------------------------------------
# Test 1: per-model within-vs-cross permutation
# ---------------------------------------------------------------------------

def perm_within_cross(M_arr, domains, n_perm=N_PERM, seed=SEED):
    n = M_arr.shape[0]
    finite = np.isfinite(M_arr) & ~np.eye(n, dtype=bool)
    same = (domains[:, None] == domains[None, :]) & finite
    diff = (domains[:, None] != domains[None, :]) & finite
    w = float(M_arr[same].mean())
    c = float(M_arr[diff].mean())
    real = w - c
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for k in range(n_perm):
        perm = rng.permutation(domains)
        s = (perm[:, None] == perm[None, :]) & finite
        d = (perm[:, None] != perm[None, :]) & finite
        null[k] = float(M_arr[s].mean() - M_arr[d].mean())
    p = float(((null >= real).sum() + 1) / (n_perm + 1))
    return w, c, p


def run_test1(pct):
    """Main-result version: permutation test on the model-averaged overlap matrix.
    Returns (within, cross, p, n_tasks) — single values, matching the paper."""
    path = os.path.join(ROOT, "results/average/overlap",
                        f"positive_{pct_token(pct)}pct_overlap.csv")
    M = pd.read_csv(path, index_col=0).values.astype(float)
    df = pd.read_csv(path, index_col=0)
    domains = np.array([l.split("/", 1)[0] for l in df.index])
    w, c, p = perm_within_cross(M, domains)
    return w, c, p, M.shape[0]


# ---------------------------------------------------------------------------
# Test 2: ARI on averaged overlap matrix (mirrors test_ablation_ari.py)
# ---------------------------------------------------------------------------

def run_test2(pct, k=ARI_K, n_clusters=4):
    """Cluster-recovery + label-aware structure metrics on the averaged matrix.

    Returns dict with:
      ari, nmi, ami, vmeas    -- compare hierarchical-cluster labels to domains
      silhouette              -- direct: are same-domain tasks closer than cross-domain?
      n                       -- number of tasks
    The first 4 require recovering discrete clusters; silhouette does not.
    """
    path = os.path.join(ROOT, "results/average/overlap",
                        f"positive_{pct_token(pct)}pct_overlap.csv")
    df = pd.read_csv(path, index_col=0)
    M = np.nan_to_num(df.values.astype(float), nan=0.0)
    labels = list(df.index)
    domains = [l.split("/", 1)[0] for l in labels]
    n = len(labels)

    M_sym = (M + M.T) / 2.0
    np.fill_diagonal(M_sym, 0)

    # kNN-based distance (mirrors test_ablation_ari.py)
    knn = np.zeros_like(M_sym)
    for i in range(n):
        top = np.argsort(M_sym[i])[::-1][:k]
        for j in top:
            knn[i, j] = M_sym[i, j]
            knn[j, i] = M_sym[j, i]
    dist_knn = 1.0 - knn
    dist_knn[knn == 0] = 1.0
    np.fill_diagonal(dist_knn, 0)

    Z = linkage(squareform(dist_knn), method="average")
    pred = fcluster(Z, t=n_clusters, criterion="maxclust")

    # Silhouette: direct on (1 - overlap) without kNN sparsification
    # — measures "are same-domain tasks closer than cross-domain ones"
    dist_full = 1.0 - M_sym
    np.fill_diagonal(dist_full, 0)
    # Force symmetry/non-negativity for sklearn
    dist_full = np.clip((dist_full + dist_full.T) / 2.0, 0.0, None)
    sil = float(silhouette_score(dist_full, domains, metric="precomputed"))

    return {
        "ari":        float(adjusted_rand_score(domains, pred)),
        "nmi":        float(normalized_mutual_info_score(domains, pred)),
        "ami":        float(adjusted_mutual_info_score(domains, pred)),
        "vmeas":      float(v_measure_score(domains, pred)),
        "silhouette": sil,
        "n":          n,
    }


# ---------------------------------------------------------------------------
# Test 3: pairwise Kendall's tau across models on common tasks
# ---------------------------------------------------------------------------

def run_test3(pct):
    matrices = {}
    for short, mdir, _ in MODELS:
        path = os.path.join(ROOT, "results", mdir,
                            f"overlap/positive_{pct_token(pct)}pct_overlap.csv")
        df = pd.read_csv(path, index_col=0)
        matrices[short] = (list(df.index), df.values.astype(float))

    common = sorted(set.intersection(*[set(rows) for rows, _ in matrices.values()]))
    vecs = {}
    for name, (rows, M) in matrices.items():
        idx = [rows.index(t) for t in common]
        sub = M[np.ix_(idx, idx)]
        # Overlap is symmetric; use upper triangle (off-diagonal)
        iu = np.triu_indices(len(common), k=1)
        vecs[name] = sub[iu]

    names = list(matrices.keys())
    taus, ps = [], []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            v1, v2 = vecs[names[i]], vecs[names[j]]
            ok = ~(np.isnan(v1) | np.isnan(v2))
            tau, p = kendalltau(v1[ok], v2[ok])
            taus.append(tau)
            ps.append(p)
    taus = np.array(taus)
    ps   = np.array(ps)
    return taus.mean(), taus.std(ddof=1), taus.min(), taus.max(), len(taus), float(ps.max())


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

summary = []
print(f"Sweeping thresholds: {PCTS}\n")
for pct in PCTS:
    print(f"=== top {pct}% ===")
    w, c, p, n_t1 = run_test1(pct)
    ratio = w / c if c > 0 else float("nan")
    print(f"  Test 1 (perm on averaged matrix): "
          f"within {w*100:.2f}%  cross {c*100:.2f}%  ratio {ratio:.2f}×  p = {p:.2g}  (n_tasks={n_t1})")

    t2 = run_test2(pct)
    print(f"  Test 2 (cluster recovery on averaged):")
    print(f"    ARI={t2['ari']:.3f}  NMI={t2['nmi']:.3f}  AMI={t2['ami']:.3f}  "
          f"V-meas={t2['vmeas']:.3f}  Silhouette={t2['silhouette']:.3f}  "
          f"(n_tasks={t2['n']})")

    tau_mean, tau_std, tau_min, tau_max, n_pairs, p_max = run_test3(pct)
    print(f"  Test 3 (pairwise Kendall):        tau = {tau_mean:.3f}±{tau_std:.3f}  "
          f"range [{tau_min:.3f}, {tau_max:.3f}]  ({n_pairs} pairs, max p = {p_max:.2g})")

    summary.append({
        "pct": pct,
        "within_pct":   w * 100,
        "cross_pct":    c * 100,
        "ratio":        ratio,
        "perm_p":       p,
        "n_tasks":      n_t1,
        "ari":          t2["ari"],
        "nmi":          t2["nmi"],
        "ami":          t2["ami"],
        "v_measure":    t2["vmeas"],
        "silhouette":   t2["silhouette"],
        "kendall_mean": tau_mean,
        "kendall_sd":   tau_std,
        "kendall_min":  tau_min,
        "kendall_max":  tau_max,
        "kendall_p_max": p_max,
        "n_pairs":      n_pairs,
    })

df = pd.DataFrame(summary)
os.makedirs(os.path.join(ROOT, "docs"), exist_ok=True)
csv_path = os.path.join(ROOT, "docs/threshold_sweep_summary.csv")
df.to_csv(csv_path, index=False, float_format="%.4f")
print(f"\nSaved CSV: {csv_path}")


def fmt_p(p, n=N_PERM):
    return f"<{1.0/n:.0e}" if p < 1.0 / n else f"{p:.2g}"


md = []
md.append("| Top-X% | Within (%) | Cross (%) | W/C ratio | Perm p | "
          "ARI | NMI | AMI | V-meas | Silhouette | "
          "Kendall τ (mean ± SD, 15 pairs) |")
md.append("|---|---|---|---|---|---|---|---|---|---|---|")
for r in summary:
    md.append(
        "| {pct}% | {w:.2f} | {c:.2f} | {rt:.2f}× | {pp} | "
        "{ari:.3f} | {nmi:.3f} | {ami:.3f} | {vm:.3f} | {sil:.3f} | "
        "{km:.3f} ± {ks:.3f} |".format(
            pct=r["pct"],
            w=r["within_pct"], c=r["cross_pct"], rt=r["ratio"],
            pp=fmt_p(r["perm_p"]),
            ari=r["ari"], nmi=r["nmi"], ami=r["ami"],
            vm=r["v_measure"], sil=r["silhouette"],
            km=r["kendall_mean"], ks=r["kendall_sd"],
        )
    )

md_chunk = "\n".join(md) + "\n"
md_path = os.path.join(ROOT, "docs/threshold_sweep_summary.md")
open(md_path, "w").write(md_chunk)
print(f"Saved MD:  {md_path}\n")
print(md_chunk)
