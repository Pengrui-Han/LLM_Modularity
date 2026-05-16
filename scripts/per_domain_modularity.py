"""Per-domain within-vs-cross modularity test, on the AVERAGED matrices.

Inputs (cross-model averages, same source as the main-text 12.9% vs 3.0%):
  results/average/overlap/positive_0.1pct_overlap.csv          (46x46)
  results/average/ablation/positive_0.1pct_corrupted_accuracy.csv (46x46)

For each target domain D vs not-D:
  within_D = mean of off-diagonal cells (i,j) with both i,j in D
  cross_D  = mean of cells (i,j) with exactly one of i,j in D
  perm p   = fraction of 10,000 label-shuffles where within-cross >= real
             (shuffle full task->domain assignment, preserving domain sizes)

Both matrices are read at their own orientation; ablation is asymmetric
(source x target) and we use ALL off-diagonal cells (both (i,j) and (j,i)).
"""
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OVERLAP_CSV  = ROOT / "results/average/overlap/positive_0.1pct_overlap.csv"
ABLATION_CSV = ROOT / "results/average/ablation/positive_0.1pct_corrupted_accuracy.csv"

N_PERM = 10_000
SEED   = 42

DOMAIN_LABELS = {
    "MD":   "Multi-Demand",
    "Lan":  "Language",
    "phys": "Physical",
    "ToM":  "Theory of Mind",
}

def domain_of(task: str) -> str:
    return task.split("/", 1)[0]


def per_domain_perm(A: np.ndarray, domains: np.ndarray, target: str,
                    n_perm=N_PERM, seed=SEED):
    """For domain `target` vs not-`target`, return (within, cross, ratio, p).
    Uses ALL off-diagonal cells (works for both symmetric overlap and
    asymmetric ablation). Permutation shuffles task->domain assignment."""
    n = A.shape[0]
    off_diag = ~np.eye(n, dtype=bool)

    is_tgt = (domains == target)
    w_mask = (is_tgt[:, None] & is_tgt[None, :]) & off_diag
    c_mask = (is_tgt[:, None] ^ is_tgt[None, :])           # exactly one in D
    within_real = float(A[w_mask].mean())
    cross_real  = float(A[c_mask].mean())
    diff_real   = within_real - cross_real

    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for k in range(n_perm):
        perm = rng.permutation(domains)
        is_t = (perm == target)
        s = (is_t[:, None] & is_t[None, :]) & off_diag
        c = (is_t[:, None] ^ is_t[None, :])
        null[k] = float(A[s].mean() - A[c].mean())
    p = float(((null >= diff_real).sum() + 1) / (n_perm + 1))
    return within_real, cross_real, (within_real / cross_real if cross_real > 0 else np.nan), p


def fmt_p(p):
    return f"<{1.0/N_PERM:.0e}" if p < 1.0 / N_PERM else f"{p:.4g}"


def main():
    Mo = pd.read_csv(OVERLAP_CSV,  index_col=0)
    Ma = pd.read_csv(ABLATION_CSV, index_col=0)
    assert list(Mo.index) == list(Ma.index), "task index mismatch"
    tasks = np.array(Mo.index.tolist())
    domains = np.array([domain_of(t) for t in tasks])
    Aov, Aab = Mo.values.astype(float), Ma.values.astype(float)

    rows = []
    for d_key, d_label in DOMAIN_LABELS.items():
        n_in = int((domains == d_key).sum())
        wo, co, ro, po = per_domain_perm(Aov, domains, d_key)
        wa, ca, ra, pa = per_domain_perm(Aab, domains, d_key)
        rows.append(dict(
            domain=d_label, n_tasks=n_in,
            overlap_within=wo, overlap_cross=co,
            overlap_ratio=ro,  overlap_p=po,
            ablation_within=wa, ablation_cross=ca,
            ablation_ratio=ra,  ablation_p=pa,
        ))
        print(f"  {d_label:<14s} (n={n_in})")
        print(f"     Overlap   within={wo*100:5.2f}%  cross={co*100:5.2f}%  "
              f"W/C={ro:5.2f}x  p={fmt_p(po)}")
        print(f"     Ablation  within={wa:5.3f}    cross={ca:5.3f}     "
              f"W/C={ra:5.2f}x  p={fmt_p(pa)}")

    df = pd.DataFrame(rows)
    csv_out = ROOT / "results" / "per_domain_modularity_avgmatrix.csv"
    df.to_csv(csv_out, index=False, float_format="%.6f")
    print(f"\nWrote {csv_out}")

    # Markdown chunk for SI
    md = []
    md.append("| Domain | n tasks | Overlap within (%) | Overlap cross (%) | W/C | Overlap perm p | Ablation within (Δacc) | Ablation cross (Δacc) | W/C | Ablation perm p |")
    md.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        md.append(
            f"| {r['domain']} | {r['n_tasks']} "
            f"| {r['overlap_within']*100:.1f} "
            f"| {r['overlap_cross']*100:.1f} "
            f"| {r['overlap_ratio']:.1f}× "
            f"| {fmt_p(r['overlap_p'])} "
            f"| {r['ablation_within']:.3f} "
            f"| {r['ablation_cross']:.3f} "
            f"| {r['ablation_ratio']:.1f}× "
            f"| {fmt_p(r['ablation_p'])} |"
        )
    md_out = ROOT / "docs" / "per_domain_modularity_summary.md"
    md_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.write_text("\n".join(md) + "\n")
    print(f"Wrote {md_out}\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
