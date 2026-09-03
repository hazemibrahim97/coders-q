from __future__ import annotations
import argparse, hashlib, json, os, sys, time
import numpy as np
import duckdb

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import random_impact as RI
import qmodel as QM

WORK_REACH = os.environ.get("CODERSQ_WORK_REACH",
                            "/scratch/$USER/codersq/impact/joined/work_reach.parquet")
SPLIT_SALT = "codersq-2026-07-11-discovery-holdout-v1"
G_MIN_SHARE = 0.10
SPAN_MIN_MO = 24

def log(*a): print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)

CREDIT_RULES = ("R-share", "R-lead", "R-core")
WS           = (24, 36, 60, 120)
N_MINS       = (3, 5, 10)
PROXIES      = ("reach2", "indeg")
PRIMARY = dict(credit="R-share", W=60, n_min=5, proxy="reach2")

def split_tag(login):
    h = hashlib.md5((SPLIT_SALT + "|" + login).encode()).hexdigest()
    return "discovery" if (int(h[:8], 16) & 1) == 0 else "holdout"

def build_sequences(con, half, credit, W, n_min, proxy, completed=True):
    impact_col = "c_reach2_sum" if proxy == "reach2" else "c_indeg_sum"
    if credit == "R-share":
        g_expr, incl = "credit_share", "credit_share >= %f" % G_MIN_SHARE
    elif credit == "R-lead":
        g_expr, incl = "1.0", "is_lead"
    else:
        g_expr, incl = "1.0", "is_core"
    cens = "AND NOT extrap_right" if completed else ""

    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE cell AS
        SELECT login, repo, first_ts, dom_eco, work_cohort_year,
               {g_expr} AS g,
               ln(1 + GREATEST({impact_col}, 0)) AS lc
        FROM read_parquet('{WORK_REACH}')
        WHERE W = {W} AND ({incl}) {cens};
    """)

    logins = [r[0] for r in con.execute("SELECT DISTINCT login FROM cell").fetchall()]
    tags = [(lg, split_tag(lg)) for lg in logins]
    con.execute("CREATE OR REPLACE TEMP TABLE splt(login VARCHAR, tag VARCHAR);")
    if tags:
        con.executemany("INSERT INTO splt VALUES (?,?)", tags)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE cellh AS
        SELECT c.* FROM cell c JOIN splt s USING(login) WHERE s.tag = '{half}';
    """)

    con.execute("""
        CREATE OR REPLACE TEMP TABLE med AS
        SELECT dom_eco, work_cohort_year, median(lc) AS cell_med
        FROM cellh GROUP BY dom_eco, work_cohort_year;
    """)
    con.execute("""
        CREATE OR REPLACE TEMP TABLE realized AS
        SELECT h.login, h.repo, h.first_ts,
               h.g * (h.lc - m.cell_med) AS L,
               (h.lc = 0) AS is_zero
        FROM cellh h JOIN med m USING(dom_eco, work_cohort_year);
    """)

    rows = con.execute(f"""
        WITH agg AS (
          SELECT login,
                 list(L ORDER BY first_ts, repo)  AS Ls,
                 count(*)                          AS n,
                 date_diff('month', min(first_ts), max(first_ts)) AS span_mo,
                 sum(CASE WHEN is_zero THEN 1 ELSE 0 END) AS n_zero
          FROM realized GROUP BY login
        )
        SELECT login, Ls, n, n_zero FROM agg
        WHERE n >= {n_min} AND span_mo >= {SPAN_MIN_MO};
    """).fetchall()
    seqs, nz_total, nworks = {}, 0, 0
    for login, Ls, n, n_zero in rows:
        seqs[login] = np.exp(np.asarray(Ls, dtype=float))
        nz_total += n_zero; nworks += n
    return seqs, dict(n_contributors=len(seqs), n_works=nworks,
                      frac_zero=(nz_total / nworks if nworks else float("nan")))

def run_cell(seqs, rng, n_perm, n_boot):
    seqlist = list(seqs.values())

    P, lengths, n_drop = RI.normalized_positions(seqlist, rng=rng, tie="random", drop_all_tie=True)
    c1 = RI.test_random_impact(P, lengths, n_bins=10, rng=rng, n_perm=n_perm) if P.size else None

    c2 = QM.decompose_autocorr(seqlist, max_lag=8, n_boot=n_boot, rng=rng)
    ac = QM.within_person_autocorr(seqlist, max_lag=8)
    fit = QM.fit_qmodel(seqlist)
    shr = QM.split_half_Q_consistency(seqlist, rng)
    return {
        "n_contributors": len(seqlist),
        "n_used_C1": int(P.size), "n_dropped_alltie_C1": int(n_drop),
        "C1": None if c1 is None else {
            "mean_P": c1["mean_P"], "null_mean_P": c1["null_mean_P"],
            "chi2": c1["chi2"], "df": c1["df"], "chi2_p": c1["chi2_p"], "perm_p": c1["perm_p"],
            "tvd": c1["tvd"], "counts": c1["counts"], "expected_counts": c1["expected_counts"],
        },
        "C2": {
            "A_stableQ_share": c2["stableQ_share_A"], "B_AR_share": c2["AR_share_B"],
            "ratio_A_over_AB": c2["persistent_stableQ_ratio"], "rho": c2["rho"],
            "A_ci": c2["A_ci"], "growth_ratio": c2["growth_ratio"],
            "stationary": c2["stationary"], "verdict_stable_Q": c2["verdict_stable_Q"],
            "autocorr_lag1to8": [round(float(x), 4) for x in ac["autocorr"][1:9]],
            "autocorr_slope": ac["slope"],
        },
        "qfit": {"sigma_q2": fit["sigma_q2_hat"], "sigma_p2": fit["sigma_p2_hat"],
                 "nbar_harmonic": fit["nbar_harmonic"]},
        "split_half_r": shr,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--half", default="discovery", choices=["discovery", "holdout"])
    ap.add_argument("--grid", default="full", choices=["full", "primary"])
    ap.add_argument("--out", default="/scratch/$USER/codersq/impact/estimates")
    ap.add_argument("--seed", type=int, default=20260711)
    args = ap.parse_args()

    if args.half == "holdout" and os.environ.get("CODERSQ_UNSEAL_HELDOUT") != "1":
        sys.exit("REFUSED: held-out unseal requires CODERSQ_UNSEAL_HELDOUT=1 (deliberate, timestamped).")

    os.makedirs(args.out, exist_ok=True)
    con = duckdb.connect()
    con.execute("PRAGMA threads=4;")
    rng = np.random.default_rng(args.seed)

    allrows = con.execute(f"SELECT DISTINCT login FROM read_parquet('{WORK_REACH}')").fetchall()
    disc = sum(1 for (lg,) in allrows if split_tag(lg) == "discovery")
    log(f"contributors total={len(allrows):,}  discovery={disc:,}  holdout={len(allrows)-disc:,}")
    log(f"RUNNING HALF = {args.half}  (grid={args.grid})")

    if args.grid == "primary":
        grid = [PRIMARY]
    else:
        grid = [dict(credit=c, W=w, n_min=nm, proxy=px)
                for c in CREDIT_RULES for w in WS for nm in N_MINS for px in PROXIES]

    results = []
    ckpt_path = os.path.join(args.out, f"estimates_{args.half}_{args.grid}.partial.json")
    def checkpoint():
        with open(ckpt_path, "w") as f:
            json.dump({"half": args.half, "grid": args.grid, "results": results}, f, default=float)
    for i, cell in enumerate(grid):
        is_primary = (cell["credit"] == PRIMARY["credit"] and cell["W"] == PRIMARY["W"]
                      and cell["n_min"] == PRIMARY["n_min"] and cell["proxy"] == PRIMARY["proxy"])
        n_perm = 5000 if is_primary else 1500
        n_boot = 400 if is_primary else 150
        seqs, diag = build_sequences(con, args.half, cell["credit"], cell["W"], cell["n_min"], cell["proxy"])
        if diag["n_contributors"] < 30:
            log(f"[{i+1}/{len(grid)}] {cell} -> only {diag['n_contributors']} contributors, SKIP")
            results.append({**cell, "primary": is_primary, **diag, "skipped": True})
            checkpoint()
            continue
        r = run_cell(seqs, rng, n_perm, n_boot)
        rec = {**cell, "primary": is_primary, **diag, **r}
        results.append(rec)
        c1 = r["C1"]; c2 = r["C2"]
        log(f"[{i+1}/{len(grid)}] {cell} N={diag['n_contributors']:,} fz={diag['frac_zero']:.2f} "
            f"| C1 meanP={c1['mean_P']:.3f} vs null {c1['null_mean_P']:.3f} perm_p={c1['perm_p']} "
            f"| C2 A={c2['A_stableQ_share']:.3f} ratio={c2['ratio_A_over_AB']:.2f} "
            f"stat={c2['stationary']} verdict={c2['verdict_stable_Q']}")
        checkpoint()

    def collect_p(key):
        idx, ps = [], []
        for j, r in enumerate(results):
            if r.get("skipped"): continue
            p = (r["C1"]["perm_p"] if key == "C1" else None)
            if key == "C1" and p is None: p = r["C1"]["chi2_p"]
            if p is not None: idx.append(j); ps.append(p)
        return idx, np.asarray(ps, dtype=float)
    idx1, p1 = collect_p("C1")
    if p1.size:
        rej1, adj1 = RI.bh_fdr(p1, alpha=0.05)
        for j, rj, aj in zip(idx1, rej1, adj1):
            results[j]["C1"]["bh_adj_p"] = float(aj); results[j]["C1"]["bh_reject"] = bool(rj)

    out = {
        "half": args.half, "grid": args.grid, "seed": args.seed,
        "split_salt": SPLIT_SALT, "primary_spec": PRIMARY,
        "n_contributors_total": len(allrows), "n_discovery": disc, "n_holdout": len(allrows) - disc,
        "results": results,
    }
    outp = os.path.join(args.out, f"estimates_{args.half}_{args.grid}.json")
    with open(outp, "w") as f: json.dump(out, f, indent=2, default=float)
    log(f"wrote {outp}  ({len(results)} cells)")

    prim = [r for r in results if r.get("primary") and not r.get("skipped")]
    if prim:
        r = prim[0]; c1 = r["C1"]; c2 = r["C2"]
        print("\n" + "=" * 72)
        print(f"HEADLINE CANDIDATE (discovery, primary spec {PRIMARY}):")
        print(f"  N={r['n_contributors']:,} contributors  frac_zero={r['frac_zero']:.3f}")
        print(f"  C1 random-impact: mean_P={c1['mean_P']:.3f} (null {c1['null_mean_P']:.3f})  "
              f"chi2_p={c1['chi2_p']:.2e}  perm_p={c1['perm_p']}  TVD={c1['tvd']:.3f}")
        print(f"     decile counts obs={c1['counts']}")
        print(f"                   exp={c1['expected_counts']}")
        print(f"  C2 stable-Φ: A={c2['A_stableQ_share']:.3f} (CI {c2['A_ci']})  "
              f"B={c2['B_AR_share']:.3f}  A/(A+B)={c2['ratio_A_over_AB']:.2f}  rho={c2['rho']:.2f}")
        print(f"     growth_ratio={c2['growth_ratio']:.2f} stationary={c2['stationary']}  "
              f"VERDICT stable_Q={c2['verdict_stable_Q']}")
        print(f"  split-half r={r['split_half_r']:.3f}  sigma_q2={r['qfit']['sigma_q2']:.3f}  "
              f"sigma_p2={r['qfit']['sigma_p2']:.3f}")
        print("=" * 72)
    print("\nESTIMATOR_RUN_DONE")

if __name__ == "__main__":
    main()
