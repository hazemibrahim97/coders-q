from __future__ import annotations

import numpy as np

def sample_career_lengths(M, rng, min_len=5, meanlog=2.2, sdlog=0.6):
    raw = rng.lognormal(mean=meanlog, sigma=sdlog, size=M)
    lengths = np.maximum(min_len, np.round(raw)).astype(int)
    return lengths

def simulate_qmodel(M, rng, sigma_q=np.sqrt(0.5), sigma_p=1.0, min_len=5):
    lengths = sample_career_lengths(M, rng, min_len=min_len)
    log_Q = rng.normal(0.0, sigma_q, size=M)
    sequences = []
    for i in range(M):
        n = lengths[i]
        log_r = rng.normal(0.0, sigma_p, size=n)
        sequences.append(np.exp(log_Q[i] + log_r))
    return sequences, log_Q

def inject_hot_streak(sequences, rng, frac=0.6, boost=2.5, block_frac=0.35,
                      start_conc=(1.0, 3.0)):
    a, b = start_conc
    out = []
    for seq in sequences:
        seq = np.asarray(seq, dtype=float).copy()
        n = seq.size
        if n >= 3 and rng.random() < frac:
            blk = max(1, int(round(block_frac * n)))
            max_start = n - blk
            start = int(round(rng.beta(a, b) * max_start)) if max_start > 0 else 0
            seq[start:start + blk] *= np.exp(boost)
        out.append(seq)
    return out

def fit_qmodel(sequences):
    log_means, within_vars, lengths = [], [], []
    for seq in sequences:
        lc = np.log(np.asarray(seq, dtype=float))
        n = lc.size
        log_means.append(lc.mean())
        lengths.append(n)
        if n >= 2:
            within_vars.append(lc.var(ddof=1) * (n - 1))
    log_means = np.asarray(log_means)
    lengths = np.asarray(lengths, dtype=float)

    dof = np.sum(lengths - 1)
    sigma_p2 = float(np.sum(within_vars) / dof) if dof > 0 else float("nan")

    between_var = float(np.var(log_means, ddof=1))
    nbar_h = float(len(lengths) / np.sum(1.0 / lengths))
    sigma_q2 = max(0.0, between_var - sigma_p2 / nbar_h)

    return {
        "sigma_q2_hat": sigma_q2,
        "sigma_p2_hat": sigma_p2,
        "mu_q_hat": float(np.mean(log_means)),
        "log_Q_hat": log_means,
        "nbar_harmonic": nbar_h,
        "n_contributors": int(lengths.size),
    }

def simulate_cumulative_advantage(M, rng, rho=0.85, sigma_shock=0.5, min_len=5):
    lengths = sample_career_lengths(M, rng, min_len=min_len)
    sequences = []
    for i in range(M):
        n = lengths[i]
        x = np.empty(n)
        x[0] = rng.normal(0.0, sigma_shock)
        for t in range(1, n):
            x[t] = rho * x[t - 1] + rng.normal(0.0, sigma_shock)
        sequences.append(np.exp(x))
    return sequences

def ch4_misclassification_sim(rng, n=60_000, true_gap=0.0, base=1.0,
                              corr_strength=1.2, target_err=0.15):
    n2 = n // 2
    true_south = np.concatenate([np.ones(n2, dtype=bool), np.zeros(n - n2, dtype=bool)])
    impact = base + true_gap * true_south.astype(float) + rng.normal(0, 1.0, size=n)
    z = (impact - impact.mean()) / impact.std()

    def delta(mask_south, w=None):
        if w is None:
            return float(impact[mask_south].mean() - impact[~mask_south].mean())

        ws, wn = w, 1.0 - w
        ms = np.sum(impact * ws) / np.sum(ws)
        mn = np.sum(impact * wn) / np.sum(wn)
        return float(ms - mn)

    def realized_err(b0):
        p = 1.0 / (1.0 + np.exp(-(b0 + corr_strength * z[true_south])))
        return p.mean()
    lo, hi = -12.0, 12.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if realized_err(mid) < target_err:
            lo = mid
        else:
            hi = mid
    b0 = 0.5 * (lo + hi)

    p_flip = np.where(true_south, 1.0 / (1.0 + np.exp(-(b0 + corr_strength * z))), 0.0)
    flipped = true_south & (rng.random(n) < p_flip)
    inferred_south = np.where(flipped, False, true_south)
    delta_naive = delta(inferred_south)
    realized_marg = float(p_flip[true_south].mean())

    flip_rand = rng.random(n) < target_err
    inferred_rand = np.where(true_south & flip_rand, False,
                             np.where(~true_south & flip_rand, True, true_south))
    delta_random = delta(inferred_rand)

    s0 = true_south.mean()
    pflip_model = 1.0 / (1.0 + np.exp(-(b0 + corr_strength * z)))
    w_south = np.where(inferred_south, 1.0,
                       (s0 * pflip_model) / (s0 * pflip_model + (1.0 - s0)))
    delta_corrected = delta(None, w=w_south)

    return {
        "true_gap": float(true_gap),
        "target_err": float(target_err),
        "err_south_realized": realized_marg,
        "delta_meas_correlated_floorpass": delta_naive,
        "delta_meas_balanced_random": delta_random,
        "delta_meas_bias_corrected": delta_corrected,
    }

def within_person_autocorr(sequences, max_lag=5):
    logs = [np.log(np.asarray(s, dtype=float)) for s in sequences]
    gm = np.mean(np.concatenate(logs))
    num = np.zeros(max_lag + 1)
    cnt = np.zeros(max_lag + 1)
    for x in logs:
        x = x - gm
        n = x.size
        for lag in range(0, max_lag + 1):
            if n > lag:
                num[lag] += np.sum(x[: n - lag] * x[lag:])
                cnt[lag] += n - lag
    var = num[0] / cnt[0]
    ac = np.array([(num[lag] / cnt[lag]) / var for lag in range(max_lag + 1)])
    lags = np.arange(1, max_lag + 1)
    y = ac[1:]

    slope = float(np.polyfit(lags, y, 1)[0]) if np.all(np.isfinite(y)) else float("nan")
    decay_ratio = float(y[-1] / y[0]) if y[0] != 0 else float("nan")

    stable_Q = (abs(slope) < 0.02) and (y[0] > 0.05)
    return {
        "autocorr": ac.tolist(),
        "lags": lags.tolist(),
        "slope": slope,
        "decay_ratio": decay_ratio,
        "verdict_stable_Q": bool(stable_Q),
    }

def stationarity_test(sequences, min_pos_n=150):
    logs = [np.log(np.asarray(s, dtype=float)) for s in sequences]
    gm = np.mean(np.concatenate(logs))
    maxpos = max(x.size for x in logs)
    ts, vs = [], []
    for t in range(maxpos):
        vals = np.array([x[t] - gm for x in logs if x.size > t])
        if vals.size >= min_pos_n:
            ts.append(t)
            vs.append(float(np.var(vals)))
    if len(ts) < 4:
        return {"growth_ratio": 1.0, "slope": 0.0, "stationary": True, "n_pos": len(ts)}
    ts_a, vs_a = np.array(ts, float), np.array(vs, float)
    slope = float(np.polyfit(ts_a, vs_a, 1)[0])
    k = max(1, len(vs) // 3)
    early = float(np.mean(vs_a[:k]))
    late = float(np.mean(vs_a[-k:]))
    growth = float(late / early) if early > 0 else float("inf")
    return {
        "growth_ratio": growth,
        "slope": slope,
        "stationary": bool(growth < 1.5),
        "n_pos": len(ts),
    }

def decompose_autocorr(sequences, max_lag=8, n_boot=0, rng=None):
    def _fit(seqs):
        ac = within_person_autocorr(seqs, max_lag=max_lag)["autocorr"]
        r = np.asarray(ac[1: max_lag + 1])
        lags = np.arange(1, max_lag + 1)
        best = None

        for rho in np.linspace(0.0, 0.9, 46):
            g = rho ** lags

            X = np.column_stack([np.ones_like(g), g])
            coef, _res, _rk, _sv = np.linalg.lstsq(X, r, rcond=None)
            A, B = max(0.0, coef[0]), max(0.0, coef[1])
            sse = float(np.sum((A + B * g - r) ** 2))
            if best is None or sse < best[3]:
                best = (A, B, rho, sse)
        return best

    A, B, rho, sse = _fit(sequences)

    A_lo = A_hi = None
    if n_boot and rng is not None:
        seqs = list(sequences)
        n = len(seqs)
        boots = []
        for _ in range(n_boot):
            idx = rng.integers(0, n, size=n)
            boots.append(_fit([seqs[j] for j in idx])[0])
        A_lo, A_hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))

    persistent = A + B
    ratio = float(A / persistent) if persistent > 0 else 0.0

    st = stationarity_test(sequences)
    stable = ((A > 0.10) and (ratio > 0.20) and (A_lo is None or A_lo > 0.02)
              and st["stationary"])
    return {
        "stableQ_share_A": float(A),
        "AR_share_B": float(B),
        "persistent_stableQ_ratio": ratio,
        "rho": float(rho),
        "A_ci": None if A_lo is None else [A_lo, A_hi],
        "sse": float(sse),
        "growth_ratio": st["growth_ratio"],
        "stationary": st["stationary"],
        "verdict_stable_Q": bool(stable),
    }

def split_half_Q_consistency(sequences, rng):
    a_means, b_means = [], []
    for seq in sequences:
        lc = np.log(np.asarray(seq, dtype=float))
        n = lc.size
        if n < 4:
            continue
        idx = rng.permutation(n)
        half = n // 2
        a_means.append(lc[idx[:half]].mean())
        b_means.append(lc[idx[half:2 * half]].mean())
    if len(a_means) < 3:
        return float("nan")
    return float(np.corrcoef(a_means, b_means)[0, 1])
