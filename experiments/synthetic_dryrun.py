from __future__ import annotations

import sys

import numpy as np

from qmodel import (
    ch4_misclassification_sim,
    decompose_autocorr,
    fit_qmodel,
    inject_hot_streak,
    sample_career_lengths,
    simulate_cumulative_advantage,
    simulate_qmodel,
)
from random_impact import bh_fdr, normalized_positions, test_random_impact

M = 5_000
SIGMA_Q = np.sqrt(0.5)
SIGMA_P = 1.0
SEED = 20260709
ALPHA = 0.05

def _fmt_hist(counts):
    return " ".join(f"{c:4d}" for c in counts)

def ch1(seqs, rng, tie="random"):
    P, L, dropped = normalized_positions(seqs, rng=rng, tie=tie)
    res = test_random_impact(P, L, n_bins=10, rng=rng)
    res["n_dropped"] = dropped
    return res

def main():
    rng = np.random.default_rng(SEED)
    print("=" * 72)
    print("The Coder's Q — synthetic dry-run (offline, no external calls)")
    print(f"M={M}  sigma_q^2={SIGMA_Q**2:.2f}  sigma_p^2={SIGMA_P**2:.2f}  seed={SEED}")
    print("=" * 72)

    checks = {}

    seq_A, logQ_true = simulate_qmodel(M, rng, sigma_q=SIGMA_Q, sigma_p=SIGMA_P)
    res_A = ch1(seq_A, rng)

    seq_B = inject_hot_streak(seq_A, rng, frac=0.6, boost=2.5, block_frac=0.35)
    res_B = ch1(seq_B, rng)

    _, adj = bh_fdr([res_A["chi2_p"], res_B["chi2_p"]], alpha=ALPHA)
    adj_A, adj_B = float(adj[0]), float(adj[1])

    print("\n[Regime A — pure Q-model]  expect P_i ~ shuffle null")
    print(f"  observed  (10 deciles): {_fmt_hist(res_A['counts'])}")
    print(f"  null E[.] (10 deciles): {_fmt_hist([round(x) for x in res_A['expected_counts']])}")
    print(f"  chi2={res_A['chi2']:.2f}  BH-adj p={adj_A:.3f}  perm p={res_A['perm_p']:.3f}  "
          f"mean P={res_A['mean_P']:.3f} (null {res_A['null_mean_P']:.3f})")
    checks["A: P_i ~ null (BH-adj p > alpha)"] = adj_A > ALPHA
    checks["A: mean P ~ null mean (|diff|<0.02)"] = abs(res_A["mean_P"] - res_A["null_mean_P"]) < 0.02

    print("\n[Regime B — planted early hot streak]  expect DEPARTS from null (early)")
    print(f"  observed  (10 deciles): {_fmt_hist(res_B['counts'])}")
    print(f"  chi2={res_B['chi2']:.2f}  BH-adj p={adj_B:.3g}  perm p={res_B['perm_p']:.3g}  "
          f"mean P={res_B['mean_P']:.3f} (null {res_B['null_mean_P']:.3f})")
    checks["B: P_i departs from null (BH-adj p < alpha)"] = adj_B < ALPHA
    checks["B: hot streak pulls max early (mean P < null-0.05)"] = \
        res_B["mean_P"] < res_B["null_mean_P"] - 0.05

    lengths_C = np.array([len(s) for s in seq_A])
    seq_C = [rng.poisson(0.4, size=n).astype(float) for n in lengths_C]
    res_C = ch1(seq_C, rng, tie="random")
    res_C_first = ch1(seq_C, rng, tie="first")
    _, adjc = bh_fdr([res_C["chi2_p"]], alpha=ALPHA)
    adj_C = float(adjc[0])
    print("\n[Regime C — zero-inflated Poisson(0.4)]  expect P_i ~ null with tie=random")
    print(f"  dropped all-tie careers: {res_C['n_dropped']}  (kept {res_C['n']})")
    print(f"  tie=random: chi2={res_C['chi2']:.2f}  BH-adj p={adj_C:.3f}  "
          f"mean P={res_C['mean_P']:.3f} (null {res_C['null_mean_P']:.3f})")
    print(f"  tie=first : chi2={res_C_first['chi2']:.2f}  raw p={res_C_first['chi2_p']:.3g}  "
          f"mean P={res_C_first['mean_P']:.3f}  <- OLD estimator: spurious early departure")
    checks["C: tie=random calibrated on ties (BH-adj p > alpha)"] = adj_C > ALPHA
    checks["C: tie=first IS biased (raw p < alpha) — regression guard"] = \
        res_C_first["chi2_p"] < ALPHA

    fit = fit_qmodel(seq_A)
    r_Qhat = float(np.corrcoef(fit["log_Q_hat"], logQ_true)[0, 1])
    dec_A = decompose_autocorr(seq_A, max_lag=8, n_boot=120, rng=rng)
    print("\n[Ch2 on Regime A — pure stable Q]  expect A dominant (true share ~0.33)")
    print(f"  sigma_q^2 recovered={fit['sigma_q2_hat']:.3f} (true {SIGMA_Q**2:.2f})  "
          f"corr(Q_hat,trueQ)={r_Qhat:.3f}")
    print(f"  A(stableQ share)={dec_A['stableQ_share_A']:.3f} CI={[round(x,2) for x in dec_A['A_ci']]}"
          f"  B(AR)={dec_A['AR_share_B']:.3f}  ratio={dec_A['persistent_stableQ_ratio']:.2f}"
          f"  -> stableQ={dec_A['verdict_stable_Q']}")
    checks["Ch2-A: recovers sigma_q^2 (within 20%)"] = \
        abs(fit["sigma_q2_hat"] - SIGMA_Q**2) / SIGMA_Q**2 < 0.20
    checks["Ch2-A: Q_hat tracks true Q (r > 0.6)"] = r_Qhat > 0.6
    checks["Ch2-A: decomposition affirms stable Q"] = dec_A["verdict_stable_Q"]

    seq_D = simulate_cumulative_advantage(M, rng, rho=0.85, sigma_shock=0.5)
    fit_D = fit_qmodel(seq_D)
    dec_D = decompose_autocorr(seq_D, max_lag=8, n_boot=120, rng=rng)
    print("\n[Regime D — cumulative advantage, TRUE sigma_q^2 = 0]  MoM fooled; decomposition not")
    print(f"  MoM (WRONG): sigma_q2_hat={fit_D['sigma_q2_hat']:.3f}  <- would falsely read as stable Q")
    print(f"  A={dec_D['stableQ_share_A']:.3f}  B(AR)={dec_D['AR_share_B']:.3f}  "
          f"ratio={dec_D['persistent_stableQ_ratio']:.2f}  rho={dec_D['rho']:.2f}"
          f"  -> stableQ={dec_D['verdict_stable_Q']}")
    checks["D: MoM IS fooled (sigma_q2_hat > 0.1)"] = fit_D["sigma_q2_hat"] > 0.1
    checks["D: decomposition NOT fooled (verdict NOT stable Q)"] = not dec_D["verdict_stable_Q"]

    lengths_E = sample_career_lengths(M, rng, min_len=5)
    seq_E = []
    for n in lengths_E:
        x = np.empty(n); x[0] = rng.normal(0, 0.5)
        for t in range(1, n):
            x[t] = 0.85 * x[t - 1] + rng.normal(0, 0.5)
        seq_E.append(np.exp(x + rng.normal(0, 2.0, size=n)))
    dec_E = decompose_autocorr(seq_E, max_lag=8, n_boot=120, rng=rng)
    print("\n[Regime E — cum-adv + measurement error (meas_sd=2.0)]  the F1 attack")
    print(f"  A={dec_E['stableQ_share_A']:.3f}  B={dec_E['AR_share_B']:.3f}  "
          f"ratio={dec_E['persistent_stableQ_ratio']:.2f}  -> stableQ={dec_E['verdict_stable_Q']}"
          f"   (OLD slope gate flipped to TRUE here)")
    checks["E: decomposition survives measurement error (NOT stable Q)"] = not dec_E["verdict_stable_Q"]

    seq_F = []
    logQ_F = rng.normal(0, np.sqrt(0.7), size=M)
    lengths_F = sample_career_lengths(M, rng, min_len=5)
    for i, n in enumerate(lengths_F):
        x = np.empty(n); x[0] = rng.normal(0, 0.5)
        for t in range(1, n):
            x[t] = 0.85 * x[t - 1] + rng.normal(0, 0.5)
        seq_F.append(np.exp(logQ_F[i] + x))
    dec_F = decompose_autocorr(seq_F, max_lag=8, n_boot=120, rng=rng)
    print("\n[Regime F — stable Q + AR(1) both present]  expect BOTH shares > 0 (no knife-edge)")
    print(f"  A={dec_F['stableQ_share_A']:.3f}  B={dec_F['AR_share_B']:.3f}  "
          f"ratio={dec_F['persistent_stableQ_ratio']:.2f}  -> stableQ={dec_F['verdict_stable_Q']}")
    checks["F: affirms stable Q when Q+AR coexist (A>0.1 and B>0.1)"] = \
        dec_F["stableQ_share_A"] > 0.10 and dec_F["AR_share_B"] > 0.10 and dec_F["verdict_stable_Q"]

    seq_H = []
    lengths_H = sample_career_lengths(M, rng, min_len=5)
    for n in lengths_H:
        x = np.empty(n); x[0] = rng.normal(0, 0.5)
        for t in range(1, n):
            x[t] = 1.0 * x[t - 1] + rng.normal(0, 0.5)
        seq_H.append(np.exp(x))
    dec_H = decompose_autocorr(seq_H, max_lag=8, n_boot=0, rng=rng)
    print("\n[Regime H — pure random walk (rho=1.0), NO person factor]  must REJECT stable Phi")
    print(f"  A={dec_H['stableQ_share_A']:.3f}  growth_ratio={dec_H['growth_ratio']:.2f}  "
          f"stationary={dec_H['stationary']}  -> stableQ={dec_H['verdict_stable_Q']}")
    checks["H: random walk rejected by stationarity gate (NOT stable Phi)"] = \
        not dec_H["verdict_stable_Q"]

    seq_I = []
    lengths_I = sample_career_lengths(M, rng, min_len=5)
    for n in lengths_I:
        phi = np.cumsum(rng.normal(0, 0.30, size=n)) + rng.normal(0, 0.3)
        seq_I.append(np.exp(phi + rng.normal(0, 0.5, size=n)))
    dec_I = decompose_autocorr(seq_I, max_lag=8, n_boot=0, rng=rng)
    print("[Regime I — drifting person level Phi_i(t)]  must REJECT stable Phi")
    print(f"  A={dec_I['stableQ_share_A']:.3f}  growth_ratio={dec_I['growth_ratio']:.2f}  "
          f"stationary={dec_I['stationary']}  -> stableQ={dec_I['verdict_stable_Q']}")
    checks["I: drifting Phi rejected by stationarity gate (NOT stable Phi)"] = \
        not dec_I["verdict_stable_Q"]

    checks["A: stable Phi is stationary (passes gate)"] = dec_A["stationary"]

    g = ch4_misclassification_sim(rng, n=60_000, true_gap=0.0, corr_strength=1.2, target_err=0.15)
    print("\n[Regime G — Ch4 detector artifact, EQUAL work; marginal err fixed at floor-passing 0.15]")
    print(f"  realized South->North err={g['err_south_realized']:.2f} (floor-passing, bal.acc~0.85)")
    print(f"  outcome-correlated (floor-PASSING): Delta_meas={g['delta_meas_correlated_floorpass']:+.3f}"
          f"  <- STILL MANUFACTURED (floor caps, does NOT neutralize)")
    print(f"  balanced random (never the threat):  Delta_meas={g['delta_meas_balanced_random']:+.3f}")
    print(f"  flip-model BIAS-CORRECTED:           Delta_meas={g['delta_meas_bias_corrected']:+.3f}"
          f"  <- the real fix recovers ~0")
    checks["G: floor-passing correlated error STILL manufactures a gap (|Delta|>0.1)"] = \
        abs(g["delta_meas_correlated_floorpass"]) > 0.1
    checks["G: balanced random error does not (|Delta|<0.03)"] = \
        abs(g["delta_meas_balanced_random"]) < 0.03
    checks["G: confusion/flip-model bias-correction recovers ~0 (|Delta|<0.04)"] = \
        abs(g["delta_meas_bias_corrected"]) < 0.04

    print("\n" + "-" * 72)
    all_pass = True
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        all_pass = all_pass and ok
    print("-" * 72)
    print("RESULT:", "ALL CHECKS PASSED" if all_pass else "FAILURES PRESENT")
    return 0 if all_pass else 1

if __name__ == "__main__":
    sys.exit(main())
