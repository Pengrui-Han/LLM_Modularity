"""Permutation test for the main-text ARI = 0.78.

Loads results/average/overlap/positive_0.1pct_overlap.csv, runs the canonical
clustering pipeline (symmetrize -> kNN k=5 -> hierarchical average linkage ->
4-cluster cut), computes ARI against true domain labels, then permutes domain
labels 10,000 times to get a null distribution. Reports observed ARI, null
mean/SD, and p-value.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import adjusted_rand_score

ROOT = Path(__file__).resolve().parent.parent
CSV  = ROOT / "results/average/overlap/positive_0.1pct_overlap.csv"
K    = 5
N_PERM = 10_000
SEED = 42


def cluster_pred(M: np.ndarray, k=K, n_clusters=4) -> np.ndarray:
    M_sym = (M + M.T) / 2.0
    np.fill_diagonal(M_sym, 0)
    knn = np.zeros_like(M_sym)
    for i in range(M_sym.shape[0]):
        top = np.argsort(M_sym[i])[::-1][:k]
        for j in top:
            knn[i, j] = M_sym[i, j]
            knn[j, i] = M_sym[j, i]
    dist = 1.0 - knn
    dist[knn == 0] = 1.0
    np.fill_diagonal(dist, 0)
    Z = linkage(squareform(dist), method="average")
    return fcluster(Z, t=n_clusters, criterion="maxclust")


def main():
    df = pd.read_csv(CSV, index_col=0)
    M = df.values.astype(float)
    domains = np.array([t.split("/", 1)[0] for t in df.index])

    pred = cluster_pred(M)
    ari_real = adjusted_rand_score(domains, pred)

    rng = np.random.default_rng(SEED)
    null = np.empty(N_PERM)
    for i in range(N_PERM):
        shuf = rng.permutation(domains)
        null[i] = adjusted_rand_score(shuf, pred)
    p = float(((null >= ari_real).sum() + 1) / (N_PERM + 1))

    print(f"n tasks                 : {M.shape[0]}")
    print(f"clustering pipeline     : symmetrize -> kNN k={K} -> avg-linkage -> 4-cluster")
    print(f"observed ARI            : {ari_real:.4f}")
    print(f"null ARI mean           : {null.mean():+.4f}")
    print(f"null ARI std            : {null.std():.4f}")
    print(f"max null ARI in 10k     : {null.max():.4f}")
    print(f"# null >= observed      : {int((null >= ari_real).sum())} / {N_PERM}")
    print(f"permutation p-value     : {p:.4g}  "
          f"({'<' + format(1/N_PERM, '.0e') if p < 1/N_PERM else format(p, '.4g')})")


if __name__ == "__main__":
    main()
