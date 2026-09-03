from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import duckdb
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import random_impact as RI
import qmodel as QM

WORK_REACH = os.environ.get(
    "CODERSQ_WORK_REACH",
    "/scratch/$USER/codersq/impact/joined/work_reach_v2.parquet",
)
SPLIT_SALT = "codersq-2026-07-11-discovery-holdout-v1"
G_MIN_SHARE = 0.10
SPAN_MIN_MO = 24
MIN_CONTRIBUTORS = 30

CREDIT_RULES = ("R-share", "R-lead", "R-core")
WS = (24, 36)
N_MINS = (3, 5, 10)
PROXIES = ("reach2", "indeg")
PRIMARY = dict(credit="R-share", W=24, n_min=5, proxy="reach2")

ERAS = {
    "real": {
        "pre": "first_ts < TIMESTAMP '2023-01-01'",
        "post": "first_ts >= TIMESTAMP '2023-01-01' AND first_ts < TIMESTAMP '2025-10-01'",
    },
    "placebo": {
        "pre": "first_ts < TIMESTAMP '2017-07-01'",
        "post": "first_ts >= TIMESTAMP '2017-07-01' AND first_ts < TIMESTAMP '2020-04-01'",
    },
}

def log(*parts):
    print(f"[{time.strftime('%H:%M:%S')}]", *parts, flush=True)

def split_tag(login):
    h = hashlib.md5((SPLIT_SALT + "|" + login).encode()).hexdigest()
    return "discovery" if (int(h[:8], 16) & 1) == 0 else "holdout"

def sql_literal(value):
    return value.replace("'", "''")

def credit_sql(credit):
    if credit == "R-share":
        return "credit_share", "credit_share >= %f" % G_MIN_SHARE
    if credit == "R-lead":
        return "1.0", "is_lead"
    if credit == "R-core":
        return "1.0", "is_core"
    raise ValueError(f"unknown credit rule: {credit}")

def impact_col(proxy):
    if proxy == "reach2":
        return "c_reach2_sum"
    if proxy == "indeg":
        return "c_indeg_sum"
    raise ValueError(f"unknown proxy: {proxy}")

def make_grid():
    return [
        dict(credit=credit, W=W, n_min=n_min, proxy=proxy)
        for credit in CREDIT_RULES
        for W in WS
        for n_min in N_MINS
        for proxy in PROXIES
    ]

def is_primary(cell):
    return (
        cell["credit"] == PRIMARY["credit"]
        and cell["W"] == PRIMARY["W"]
        and cell["n_min"] == PRIMARY["n_min"]
        and cell["proxy"] == PRIMARY["proxy"]
    )

def install_split_table(con, parquet_path):
    rows = con.execute(
        f"SELECT DISTINCT login FROM read_parquet('{parquet_path}')"
    ).fetchall()
    tags = [(login, split_tag(login)) for (login,) in rows]
    con.execute("CREATE OR REPLACE TEMP TABLE splt(login VARCHAR, tag VARCHAR);")
    if tags:
        con.executemany("INSERT INTO splt VALUES (?, ?)", tags)
    disc = sum(1 for _login, tag in tags if tag == "discovery")
    return len(tags), disc

def install_placebo_cohort(con, parquet_path):
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE placebo_cohort AS
        SELECT login
        FROM (
          SELECT login, min(first_ts) AS first_work_ts
          FROM read_parquet('{parquet_path}')
          WHERE NOT extrap_right
          GROUP BY login
        )
        WHERE first_work_ts >= TIMESTAMP '2015-01-01'
          AND first_work_ts < TIMESTAMP '2017-01-01';
        """
    )

def prepare_cell_tables(con, parquet_path, mode, half, cell):
    g_expr, incl = credit_sql(cell["credit"])
    c_col = impact_col(cell["proxy"])
    cohort_join = "JOIN placebo_cohort pc USING(login)" if mode == "placebo" else ""

    for era, era_where in ERAS[mode].items():
        con.execute(
            f"""
            CREATE OR REPLACE TEMP TABLE base_{era} AS
            SELECT wr.login, wr.repo, wr.first_ts, wr.dom_eco, wr.work_cohort_year,
                   {g_expr} AS g,
                   ln(1 + GREATEST(wr.{c_col}, 0)) AS lc,
                   wr.extrap_right
            FROM read_parquet('{parquet_path}') wr
            JOIN splt s USING(login)
            {cohort_join}
            WHERE wr.W = {int(cell["W"])}
              AND ({incl})
              AND s.tag = '{half}'
              AND ({era_where});
            """
        )
        con.execute(
            f"""
            CREATE OR REPLACE TEMP TABLE med_{era} AS
            SELECT dom_eco, work_cohort_year, median(lc) AS cell_med
            FROM base_{era}
            WHERE NOT extrap_right
            GROUP BY dom_eco, work_cohort_year;
            """
        )
        con.execute(
            f"""
            CREATE OR REPLACE TEMP TABLE realized_{era}_all AS
            SELECT b.login, b.repo, b.first_ts,
                   b.g * (b.lc - m.cell_med) AS L,
                   (b.lc = 0) AS is_zero
            FROM base_{era} b
            JOIN med_{era} m USING(dom_eco, work_cohort_year)
            WHERE NOT b.extrap_right;
            """
        )

        con.execute(
            f"""
            CREATE OR REPLACE TEMP TABLE qual_{era} AS
            WITH agg AS (
              SELECT login,
                     count(*) AS n,
                     date_diff('month', min(first_ts), max(first_ts)) AS span_mo
              FROM base_{era}
              GROUP BY login
            )
            SELECT login FROM agg
            WHERE n >= {int(cell["n_min"])} AND span_mo >= {SPAN_MIN_MO};
            """
        )

    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE incumbents AS
        SELECT pre.login
        FROM qual_pre pre
        JOIN qual_post post USING(login);
        """
    )
    for era in ERAS[mode]:
        con.execute(
            f"""
            CREATE OR REPLACE TEMP TABLE realized_{era} AS
            SELECT r.*
            FROM realized_{era}_all r
            JOIN incumbents i USING(login);
            """
        )

def era_sequences_and_diag(con, era):
    diag_row = con.execute(
        f"""
        SELECT count(*) AS n_works,
               sum(CASE WHEN is_zero THEN 1 ELSE 0 END) AS n_zero
        FROM realized_{era};
        """
    ).fetchone()
    n_works = int(diag_row[0] or 0)
    n_zero = int(diag_row[1] or 0)

    extrap_row = con.execute(
        f"""
        SELECT count(*) AS n_rows,
               sum(CASE WHEN b.extrap_right THEN 1 ELSE 0 END) AS n_extrap
        FROM base_{era} b
        JOIN incumbents i USING(login);
        """
    ).fetchone()
    n_rows = int(extrap_row[0] or 0)
    n_extrap = int(extrap_row[1] or 0)

    rows = con.execute(
        f"""
        SELECT login,
               list(L ORDER BY first_ts, repo) AS Ls,
               count(*) AS n
        FROM realized_{era}
        GROUP BY login
        ORDER BY login;
        """
    ).fetchall()
    seqs = {login: np.exp(np.asarray(Ls, dtype=float)) for login, Ls, _n in rows}
    diag = {
        "n_contributors": len(seqs),
        "n_works": n_works,
        "frac_zero": n_zero / n_works if n_works else float("nan"),
        "frac_extrap_right": n_extrap / n_rows if n_rows else float("nan"),
        "n_rows_before_censor": n_rows,
        "n_extrap_right": n_extrap,
    }
    return seqs, diag

def run_era(seqs, rng, n_perm, n_boot):
    logins = sorted(seqs)
    seqlist = [seqs[login] for login in logins]

    P, lengths, n_drop = RI.normalized_positions(
        seqlist, rng=rng, tie="random", drop_all_tie=True
    )
    c1 = RI.test_random_impact(P, lengths, n_bins=10, rng=rng, n_perm=n_perm) if P.size else None
    c2 = QM.decompose_autocorr(seqlist, max_lag=8, n_boot=n_boot, rng=rng)
    ac = QM.within_person_autocorr(seqlist, max_lag=8)
    fit = QM.fit_qmodel(seqlist)
    shr = QM.split_half_Q_consistency(seqlist, rng)
    phi_hat = {
        login: float(phi)
        for login, phi in zip(logins, np.asarray(fit["log_Q_hat"], dtype=float))
    }
    return {
        "result": {
            "n_contributors": len(seqlist),
            "n_used_C1": int(P.size),
            "n_dropped_alltie_C1": int(n_drop),
            "C1": None if c1 is None else {
                "mean_P": c1["mean_P"],
                "null_mean_P": c1["null_mean_P"],
                "chi2": c1["chi2"],
                "df": c1["df"],
                "chi2_p": c1["chi2_p"],
                "perm_p": c1["perm_p"],
                "tvd": c1["tvd"],
                "counts": c1["counts"],
                "expected_counts": c1["expected_counts"],
            },
            "C2": {
                "A_stableQ_share": c2["stableQ_share_A"],
                "B_AR_share": c2["AR_share_B"],
                "ratio_A_over_AB": c2["persistent_stableQ_ratio"],
                "rho": c2["rho"],
                "A_ci": c2["A_ci"],
                "growth_ratio": c2["growth_ratio"],
                "stationary": c2["stationary"],
                "verdict_stable_Q": c2["verdict_stable_Q"],
                "autocorr_lag1to8": [round(float(x), 4) for x in ac["autocorr"][1:9]],
                "autocorr_slope": ac["slope"],
            },
            "qfit": {
                "sigma_q2": fit["sigma_q2_hat"],
                "sigma_p2": fit["sigma_p2_hat"],
                "nbar_harmonic": fit["nbar_harmonic"],
            },
            "split_half_r": shr,
        },
        "phi_hat": phi_hat,
    }

def phi_rank_corr(pre_phi, post_phi):
    common = sorted(set(pre_phi).intersection(post_phi))
    if len(common) < 3:
        return {"rho": float("nan"), "p": float("nan"), "n": len(common)}
    pre = np.asarray([pre_phi[login] for login in common], dtype=float)
    post = np.asarray([post_phi[login] for login in common], dtype=float)
    if np.all(pre == pre[0]) or np.all(post == post[0]):
        return {"rho": float("nan"), "p": float("nan"), "n": len(common)}
    rho, p = stats.spearmanr(pre, post)
    return {"rho": float(rho), "p": float(p), "n": len(common)}

def checkpoint(path, payload):
    with open(path, "w") as f:
        json.dump(payload, f, default=float)

def add_bh_fdr(results):
    locs, pvals = [], []
    for i, rec in enumerate(results):
        if rec.get("skipped"):
            continue
        for era in ("pre", "post"):
            c1 = rec["per_era"][era].get("C1")
            if c1 is None:
                continue
            p = c1["perm_p"] if c1["perm_p"] is not None else c1["chi2_p"]
            if p is not None:
                locs.append((i, era))
                pvals.append(p)
    if not pvals:
        return
    rejected, adj = RI.bh_fdr(np.asarray(pvals, dtype=float), alpha=0.05)
    for (i, era), rj, aj in zip(locs, rejected, adj):
        results[i]["per_era"][era]["C1"]["bh_adj_p"] = float(aj)
        results[i]["per_era"][era]["C1"]["bh_reject"] = bool(rj)

def headline_summary(results):
    prim = [r for r in results if r.get("primary") and not r.get("skipped")]
    if not prim:
        return
    r = prim[0]
    print("\n" + "=" * 72)
    print(f"CH3 PRIMARY PRE/POST CELL ({PRIMARY}):")
    print(f"  incumbents={r['n_incumbents']:,}  phi_rank_rho={r['phi_rank_corr']['rho']:.3f}")
    for era in ("pre", "post"):
        e = r["per_era"][era]
        c1, c2 = e["C1"], e["C2"]
        print(
            f"  {era}: works={e['n_works']:,} frac_zero={e['frac_zero']:.3f} "
            f"frac_extrap={e['frac_extrap_right']:.3f}"
        )
        if c1 is None:
            print("    C1 unavailable: no non-all-tie careers after tie guard")
        else:
            print(
                f"    C1 mean_P={c1['mean_P']:.3f} null={c1['null_mean_P']:.3f} "
                f"perm_p={c1['perm_p']} TVD={c1['tvd']:.3f}"
            )
        print(
            f"    C2 A={c2['A_stableQ_share']:.3f} A/(A+B)={c2['ratio_A_over_AB']:.2f} "
            f"rho={c2['rho']:.2f} stable_Q={c2['verdict_stable_Q']} "
            f"split_half_r={e['split_half_r']:.3f}"
        )
    print("=" * 72)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="real", choices=["real", "placebo"])
    ap.add_argument("--half", default="discovery", choices=["discovery", "holdout"])
    ap.add_argument("--out", default="/scratch/$USER/codersq/impact/estimates_v2")
    ap.add_argument("--seed", type=int, default=20260711)
    args = ap.parse_args()

    if args.half == "holdout" and os.environ.get("CODERSQ_UNSEAL_HELDOUT") != "1":
        sys.exit("REFUSED: held-out unseal requires CODERSQ_UNSEAL_HELDOUT=1 (deliberate, timestamped).")

    os.makedirs(args.out, exist_ok=True)
    parquet_path = sql_literal(WORK_REACH)
    con = duckdb.connect()
    con.execute("PRAGMA threads=4;")
    rng = np.random.default_rng(args.seed)

    if args.mode == "placebo":
        install_placebo_cohort(con, parquet_path)
    n_total, n_discovery = install_split_table(con, parquet_path)
    log(
        f"contributors total={n_total:,}  discovery={n_discovery:,}  "
        f"holdout={n_total - n_discovery:,}"
    )
    log(f"RUNNING MODE = {args.mode}  HALF = {args.half}")

    grid = make_grid()
    results = []
    partial_path = os.path.join(args.out, f"prepost_{args.mode}_{args.half}.partial.json")
    final_path = os.path.join(args.out, f"prepost_{args.mode}_{args.half}.json")

    def partial_payload():
        return {
            "mode": args.mode,
            "half": args.half,
            "seed": args.seed,
            "split_salt": SPLIT_SALT,
            "primary_spec": PRIMARY,
            "n_contributors_total": n_total,
            "n_discovery": n_discovery,
            "n_holdout": n_total - n_discovery,
            "results": results,
        }

    for i, cell in enumerate(grid):
        primary = is_primary(cell)
        n_perm = 5000 if primary else 1500
        n_boot = 400 if primary else 150
        prepare_cell_tables(con, parquet_path, args.mode, args.half, cell)
        n_incumbents = int(con.execute("SELECT count(*) FROM incumbents").fetchone()[0])
        seqs_pre, diag_pre = era_sequences_and_diag(con, "pre")
        seqs_post, diag_post = era_sequences_and_diag(con, "post")
        diag_post["frac_extrap_right_post"] = diag_post["frac_extrap_right"]

        rec = {
            "cell": dict(cell),
            "credit": cell["credit"],
            "W": cell["W"],
            "n_min": cell["n_min"],
            "proxy": cell["proxy"],
            "primary": primary,
            "n_incumbents": n_incumbents,
            "per_era": {
                "pre": dict(diag_pre),
                "post": dict(diag_post),
            },
            "frac_extrap_right_post": diag_post["frac_extrap_right"],
        }
        if n_incumbents < MIN_CONTRIBUTORS:
            rec["skipped"] = True
            rec["skip_reason"] = f"only {n_incumbents} incumbents"
            results.append(rec)
            log(f"[{i+1}/{len(grid)}] {cell} -> only {n_incumbents} incumbents, SKIP")
            checkpoint(partial_path, partial_payload())
            continue

        pre = run_era(seqs_pre, rng, n_perm, n_boot)
        post = run_era(seqs_post, rng, n_perm, n_boot)
        rec["per_era"]["pre"].update(pre["result"])
        rec["per_era"]["post"].update(post["result"])
        rec["phi_rank_corr"] = phi_rank_corr(pre["phi_hat"], post["phi_hat"])
        results.append(rec)

        pre_c1 = rec["per_era"]["pre"]["C1"]
        post_c1 = rec["per_era"]["post"]["C1"]
        pre_c2 = rec["per_era"]["pre"]["C2"]
        post_c2 = rec["per_era"]["post"]["C2"]
        pre_c1_msg = (
            "C1 unavailable"
            if pre_c1 is None
            else f"C1 meanP={pre_c1['mean_P']:.3f} p={pre_c1['perm_p']}"
        )
        post_c1_msg = (
            "C1 unavailable"
            if post_c1 is None
            else f"C1 meanP={post_c1['mean_P']:.3f} p={post_c1['perm_p']}"
        )
        log(
            f"[{i+1}/{len(grid)}] {cell} N={n_incumbents:,} "
            f"pre_nw={diag_pre['n_works']:,} post_nw={diag_post['n_works']:,} "
            f"| pre {pre_c1_msg} "
            f"C2 A={pre_c2['A_stableQ_share']:.3f} ratio={pre_c2['ratio_A_over_AB']:.2f} "
            f"| post {post_c1_msg} "
            f"C2 A={post_c2['A_stableQ_share']:.3f} ratio={post_c2['ratio_A_over_AB']:.2f} "
            f"| phi_rho={rec['phi_rank_corr']['rho']:.3f}"
        )
        checkpoint(partial_path, partial_payload())

    add_bh_fdr(results)
    out = partial_payload()
    with open(final_path, "w") as f:
        json.dump(out, f, indent=2, default=float)
    checkpoint(partial_path, out)
    log(f"wrote {final_path}  ({len(results)} cells)")
    headline_summary(results)
    print("\nPREPOST_RUN_DONE")

if __name__ == "__main__":
    main()
