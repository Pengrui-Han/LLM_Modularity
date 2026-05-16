"""Per-model modularity summary for the SI table.

For each of the 6 models, compute on BOTH:
  (1) overlap matrix      (positive_0.1pct_overlap.csv)
  (2) ablation matrix     (positive_0.1pct_corrupted_accuracy.csv, off-diagonal,
                           any NaN cells dropped from each pair)
the within-vs-cross-domain permutation test:
  - within_mean / cross_mean
  - ratio = within_mean / cross_mean
  - permutation p (10,000 iterations) for the within - cross difference, by
    shuffling task labels and recomputing the difference each time

Outputs:
  docs/per_model_modularity_summary.csv
  docs/per_model_modularity_summary.md   (markdown table chunk for the SI)
"""
import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODELS = [
    ("Mistral-Small-24B",   "mistralai_Mistral-Small-24B-Instruct-2501",   24),
    ("Qwen2.5-32B",         "Qwen_Qwen2-5-32B-Instruct",                   32),
    ("OLMo-2-32B",          "allenai_OLMo-2-0325-32B-Instruct",            32),
    ("Llama-3.1-70B",       "meta-llama_Meta-Llama-3-1-70B-Instruct",      70),
    ("Qwen2.5-72B",         "Qwen_Qwen2-5-72B-Instruct",                   72),
    ("Mistral-Large-123B",  "mistralai_Mistral-Large-Instruct-2407",      123),
]
N_PERM = 10000
SEED = 42


def load_overlap(model_dir):
    p = os.path.join(ROOT, "results", model_dir, "overlap/positive_0.1pct_overlap.csv")
    return pd.read_csv(p, index_col=0)


def load_ablation(model_dir):
    p = os.path.join(ROOT, "results", model_dir,
                     "ablation_analysis/positive_0.1pct_corrupted_accuracy.csv")
    df = pd.read_csv(p, index_col=0)
    return df


def perm_within_cross(M_arr, domains, n_perm=N_PERM, seed=SEED, exclude_diag=True):
    """Return (within_mean, cross_mean, perm_p). M_arr can have NaNs (treated as missing).
    'within' = same-domain off-diagonal cells; 'cross' = different-domain cells.
    Permutation: shuffle the per-task domain label."""
    n = M_arr.shape[0]
    finite = np.isfinite(M_arr)
    if exclude_diag:
        finite = finite & ~np.eye(n, dtype=bool)
    same = (domains[:, None] == domains[None, :]) & finite
    diff = (domains[:, None] != domains[None, :]) & finite
    within_mean = float(M_arr[same].mean())
    cross_mean  = float(M_arr[diff].mean())
    real_diff   = within_mean - cross_mean

    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for k in range(n_perm):
        perm = rng.permutation(domains)
        s = (perm[:, None] == perm[None, :]) & finite
        d = (perm[:, None] != perm[None, :]) & finite
        null[k] = float(M_arr[s].mean() - M_arr[d].mean())
    p = float(((null >= real_diff).sum() + 1) / (n_perm + 1))
    return within_mean, cross_mean, p


# ------------- run ----------------------------------------------------------

rows = []
for short, mdir, params in MODELS:
    print(f"  {short} ...", flush=True)

    # (1) Overlap matrix (no NaNs expected)
    Mo = load_overlap(mdir)
    domains_o = np.array([l.split("/", 1)[0] for l in Mo.index])
    Aov = Mo.values.astype(float)
    w_ov, c_ov, p_ov = perm_within_cross(Aov, domains_o)

    # (2) Ablation matrix
    Mab = load_ablation(mdir)
    # rows and cols may not exactly match overlap labels; intersect
    common = [l for l in Mab.index if l in Mab.columns]
    Mab2 = Mab.loc[common, common]
    domains_a = np.array([l.split("/", 1)[0] for l in Mab2.index])
    Aab = Mab2.values.astype(float)
    w_ab, c_ab, p_ab = perm_within_cross(Aab, domains_a)

    rows.append({
        "Model": short, "Params (B)": params,
        "n_tasks_overlap": Aov.shape[0],
        "overlap_within_pct": w_ov * 100,
        "overlap_cross_pct":  c_ov * 100,
        "overlap_ratio":      (w_ov / c_ov) if c_ov > 0 else np.nan,
        "overlap_perm_p":     p_ov,
        "n_tasks_ablation":   Aab.shape[0],
        "ablation_within":    w_ab,
        "ablation_cross":     c_ab,
        "ablation_ratio":     (w_ab / c_ab) if c_ab > 0 else np.nan,
        "ablation_perm_p":    p_ab,
    })

df = pd.DataFrame(rows)

os.makedirs(os.path.join(ROOT, "docs"), exist_ok=True)
csv_path = os.path.join(ROOT, "docs/per_model_modularity_summary.csv")
df.to_csv(csv_path, index=False, float_format="%.4f")
print(f"\nSaved CSV: {csv_path}")

# Markdown table chunk for SI
def fmt_p(p):
    return f"<{1.0/N_PERM:.0e}" if p < 1.0 / N_PERM else f"{p:.4g}"

md = []
md.append("| Model | Params (B) | Overlap within (%) | Overlap cross (%) | W/C | Overlap perm p | Ablation within (Δacc) | Ablation cross (Δacc) | W/C | Ablation perm p |")
md.append("|---|---|---|---|---|---|---|---|---|---|")
for _, r in df.iterrows():
    md.append(("| {model} | {pb} | {ow:.1f} | {oc:.1f} | {orr:.1f}× | {opp} | "
               "{aw:.3f} | {ac:.3f} | {arr:.1f}× | {app} |").format(
        model=r["Model"], pb=int(r["Params (B)"]),
        ow=r["overlap_within_pct"], oc=r["overlap_cross_pct"],
        orr=r["overlap_ratio"], opp=fmt_p(r["overlap_perm_p"]),
        aw=r["ablation_within"], ac=r["ablation_cross"],
        arr=r["ablation_ratio"], app=fmt_p(r["ablation_perm_p"])))

# aggregate row
def msd(s, fmt="%.1f"): return (fmt % s.mean()) + " ± " + (fmt % s.std(ddof=1))
md.append(("| **Mean ± SD** | — | **{ow}** | **{oc}** | **{orr}** | all <{thr:.0e} | "
           "**{aw}** | **{ac}** | **{arr}** | all <{thr:.0e} |").format(
    ow=msd(df["overlap_within_pct"]),
    oc=msd(df["overlap_cross_pct"]),
    orr=msd(df["overlap_ratio"]),
    aw=msd(df["ablation_within"], "%.3f"),
    ac=msd(df["ablation_cross"],  "%.3f"),
    arr=msd(df["ablation_ratio"]),
    thr=1.0/N_PERM))

md_chunk = "\n".join(md) + "\n"
md_path = os.path.join(ROOT, "docs/per_model_modularity_summary.md")
open(md_path, "w").write(md_chunk)
print(f"Saved MD chunk: {md_path}\n")
print(md_chunk)
