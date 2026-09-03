import numpy as np
from qmodel import (within_person_autocorr, fit_qmodel, sample_career_lengths,
                    simulate_cumulative_advantage, simulate_qmodel)

SEED = 7; M = 5000

def verdict(seqs, label):
    ac = within_person_autocorr(seqs, max_lag=5)
    print(f"{label:52s} slope={ac['slope']:+.4f}  ac1={ac['autocorr'][1]:.3f}  "
          f"ac5={ac['autocorr'][5]:.3f}  stableQ={ac['verdict_stable_Q']}")
    return ac

def sim_mixed(M, rng, sigma_q, rho, sigma_shock, min_len=5):
    lengths = sample_career_lengths(M, rng, min_len=min_len)
    logQ = rng.normal(0.0, sigma_q, size=M)
    seqs = []
    for i in range(M):
        n = lengths[i]; x = np.empty(n)
        x[0] = rng.normal(0.0, sigma_shock)
        for t in range(1, n):
            x[t] = rho * x[t-1] + rng.normal(0.0, sigma_shock)
        seqs.append(np.exp(logQ[i] + x))
    return seqs

print("=== ATTACK 1: stable Q + AR(1) both present ===")
for sq, rho in [(0.0,0.85),(0.3,0.85),(0.7,0.85),(1.0,0.85),(0.5,0.5),(1.2,0.85),(2.0,0.85)]:
    rng = np.random.default_rng(SEED)
    verdict(sim_mixed(M,rng,np.sqrt(sq),rho,0.5), f"sigma_q^2={sq:.2f} rho={rho}")

print("\n=== ATTACK 2: iid measurement error on PURE cum-adv (true sigma_q^2=0) ===")
def sim_cumadv_noisy(M, rng, rho, sigma_shock, sigma_meas, min_len=5):
    lengths = sample_career_lengths(M, rng, min_len=min_len)
    seqs=[]
    for i in range(M):
        n=lengths[i]; x=np.empty(n); x[0]=rng.normal(0,sigma_shock)
        for t in range(1,n): x[t]=rho*x[t-1]+rng.normal(0,sigma_shock)
        x = x + rng.normal(0,sigma_meas,size=n)
        seqs.append(np.exp(x))
    return seqs
for sm in [0.0,0.5,1.0,1.5,2.0,3.0]:
    rng=np.random.default_rng(SEED)
    verdict(sim_cumadv_noisy(M,rng,0.85,0.5,sm), f"meas_sd={sm}")

print("\n=== ATTACK 3: pure cum-adv across rho (true sigma_q^2=0, should NOT be stableQ) ===")
for rho in [0.2,0.3,0.4,0.5,0.6,0.7]:
    rng=np.random.default_rng(SEED)
    verdict(simulate_cumulative_advantage(M,rng,rho=rho,sigma_shock=0.5), f"rho={rho}")

print("\n=== ATTACK 4: short careers (min_len small), pure cum-adv ===")
def sim_cumadv_len(M,rng,rho,ss,meanlog,sdlog,min_len):
    lengths=np.maximum(min_len,np.round(rng.lognormal(meanlog,sdlog,M))).astype(int)
    seqs=[]
    for i in range(M):
        n=lengths[i]; x=np.empty(n); x[0]=rng.normal(0,ss)
        for t in range(1,n): x[t]=rho*x[t-1]+rng.normal(0,ss)
        seqs.append(np.exp(x))
    return seqs
for ml,sl in [(1.2,0.5),(1.6,0.5),(2.2,0.6)]:
    rng=np.random.default_rng(SEED)
    Ls=np.maximum(5 if ml>2 else 3,np.round(rng.lognormal(ml,sl,M))).astype(int)
    print(f"  median career len ~ {int(np.median(Ls))}")
    rng=np.random.default_rng(SEED)
    verdict(sim_cumadv_len(M,rng,0.85,0.5,ml,sl,3), f"meanlog={ml}")
