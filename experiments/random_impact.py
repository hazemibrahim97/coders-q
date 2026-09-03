from __future__ import annotations

import numpy as np
from scipy import stats

def normalized_positions(sequences, rng=None, tie="random", drop_all_tie=True):
    if tie == "random" and rng is None:
        raise ValueError("tie='random' requires an rng (np.random.Generator)")
    P, lengths, n_dropped = [], [], 0
    for seq in sequences:
        seq = np.asarray(seq, dtype=float)
        n = seq.size
        if n < 2:
            n_dropped += 1
            continue
        mx = seq.max()
        idx = np.flatnonzero(seq == mx)
        if drop_all_tie and idx.size == n:
            n_dropped += 1
            continue
        if tie == "random":
            pos = int(rng.choice(idx)) + 1
        else:
            pos = int(idx[0]) + 1
        P.append(pos / n)
        lengths.append(n)
    return np.asarray(P), np.asarray(lengths, dtype=int), n_dropped

def _bin_edges(n_bins):
    return np.linspace(0.0, 1.0, n_bins + 1)

def null_decile_expectation(lengths, n_bins=10):
    edges = _bin_edges(n_bins)
    exp_counts = np.zeros(n_bins)
    for N in np.asarray(lengths, dtype=int):
        if N < 2:
            continue
        P = np.arange(1, N + 1) / N
        h, _ = np.histogram(np.clip(P, 1e-12, 1.0), bins=edges)
        exp_counts += h / N
    return exp_counts

def null_mean_P(lengths):
    L = np.asarray(lengths, dtype=float)
    L = L[L >= 2]
    return float(np.mean((L + 1.0) / (2.0 * L)))

def test_random_impact(P, lengths, n_bins=10, rng=None, n_perm=2000):
    P = np.asarray(P, dtype=float)
    lengths = np.asarray(lengths, dtype=int)
    n = P.size
    edges = _bin_edges(n_bins)
    obs, _ = np.histogram(np.clip(P, 1e-12, 1.0), bins=edges)
    exp = null_decile_expectation(lengths, n_bins)

    mask = exp > 0
    chi2 = float(np.sum((obs[mask] - exp[mask]) ** 2 / exp[mask]))
    df = int(mask.sum() - 1)
    chi2_p = float(stats.chi2.sf(chi2, df=df))

    nullmu = null_mean_P(lengths)
    tvd = float(0.5 * np.sum(np.abs(obs / n - exp / n)))

    perm_p = None
    if rng is not None and n_perm:
        stat_null = np.empty(n_perm)
        Lg2 = lengths[lengths >= 2]
        for t in range(n_perm):
            k = np.floor(rng.random(Lg2.size) * Lg2).astype(int) + 1
            Pn = k / Lg2
            on, _ = np.histogram(np.clip(Pn, 1e-12, 1.0), bins=edges)
            stat_null[t] = np.sum((on[mask] - exp[mask]) ** 2 / exp[mask])
        perm_p = float((np.sum(stat_null >= chi2) + 1) / (n_perm + 1))

    return {
        "n": n,
        "n_bins": n_bins,
        "counts": obs.tolist(),
        "expected_counts": [round(float(x), 1) for x in exp],
        "chi2": chi2,
        "df": df,
        "chi2_p": chi2_p,
        "perm_p": perm_p,
        "tvd": tvd,
        "mean_P": float(np.mean(P)),
        "null_mean_P": nullmu,
    }

def bh_fdr(pvals, alpha=0.05):
    p = np.asarray(pvals, dtype=float)
    m = p.size
    order = np.argsort(p)
    ranked = p[order]
    adj = ranked * m / (np.arange(1, m + 1))

    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    out = np.empty_like(adj)
    out[order] = adj
    rejected = out <= alpha
    return rejected, out
