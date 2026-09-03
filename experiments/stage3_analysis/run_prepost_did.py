from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import run_prepost as RP
import qmodel as QM

RI = RP.RI

REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_NOTE = "prereg/POSTHOC-NOTE-did-and-censoring.md"
SPEC_NOTE_PATH = REPO_ROOT / SPEC_NOTE
DEFAULT_SEED = 20260711
TOST_MARGIN = 0.10

def log(*parts):
    print(f"[{time.strftime('%H:%M:%S')}]", *parts, flush=True)

def json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

def write_json(path, payload):
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=json_default)

def month_add_expr(column, W):
    return f"{column} + INTERVAL {int(W)} MONTH"

def cell_seed(base_seed, half, cell, matched_censoring=False):
    key = "|".join(
        [
            str(int(base_seed)),
            half,
            "matched" if matched_censoring else "standard",
            str(cell["credit"]),
            str(int(cell["W"])),
            str(int(cell["n_min"])),
            str(cell["proxy"]),
        ]
    )
    h = hashlib.sha256(key.encode()).hexdigest()
    return int(h[:16], 16) % (2**32)

def is_primary(cell):
    return RP.is_primary(cell)

def ci(draws, lo, hi):
    a = np.asarray(draws, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return [float("nan"), float("nan")]
    return [float(np.percentile(a, lo)), float(np.percentile(a, hi))]

def two_sided_zero_p(draws):
    a = np.asarray(draws, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float("nan")
    return float(min(1.0, 2.0 * min(np.mean(a <= 0.0), np.mean(a >= 0.0))))

def c2_from_seqs(seqlist):
    c2 = QM.decompose_autocorr(seqlist, max_lag=8, n_boot=0)
    return {
        "A_stableQ_share": c2["stableQ_share_A"],
        "B_AR_share": c2["AR_share_B"],
        "ratio_A_over_AB": c2["persistent_stableQ_ratio"],
        "rho": c2["rho"],
        "A_ci": c2["A_ci"],
        "growth_ratio": c2["growth_ratio"],
        "stationary": c2["stationary"],
        "verdict_stable_Q": c2["verdict_stable_Q"],
    }

def c1_exact_tvd(seqlist, rng):
    P, lengths, n_drop = RI.normalized_positions(
        seqlist, rng=rng, tie="random", drop_all_tie=True
    )
    if not P.size:
        return {
            "n_used_C1": 0,
            "n_dropped_alltie_C1": int(n_drop),
            "C1": None,
            "tvd": float("nan"),
        }
    c1 = RI.test_random_impact(P, lengths, n_bins=10, rng=rng, n_perm=0)
    return {
        "n_used_C1": int(P.size),
        "n_dropped_alltie_C1": int(n_drop),
        "C1": {
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
        "tvd": c1["tvd"],
    }

def seqlist_works(seqlist):
    return int(sum(np.asarray(seq).size for seq in seqlist))

def cohort_stats_from_lists(pre_list, post_list, rng):
    pre_c2 = c2_from_seqs(pre_list)
    post_c2 = c2_from_seqs(post_list)
    pre_c1 = c1_exact_tvd(pre_list, rng)
    post_c1 = c1_exact_tvd(post_list, rng)
    A_pre = float(pre_c2["A_stableQ_share"])
    A_post = float(post_c2["A_stableQ_share"])
    T_pre = float(pre_c1["tvd"])
    T_post = float(post_c1["tvd"])
    return {
        "A_pre": A_pre,
        "A_post": A_post,
        "delta_A": A_post - A_pre,
        "TVD_pre": T_pre,
        "TVD_post": T_post,
        "delta_TVD": T_post - T_pre,
        "per_era": {
            "pre": {
                "n_contributors": len(pre_list),
                "n_works": seqlist_works(pre_list),
                "C1": pre_c1["C1"],
                "n_used_C1": pre_c1["n_used_C1"],
                "n_dropped_alltie_C1": pre_c1["n_dropped_alltie_C1"],
                "C2": pre_c2,
            },
            "post": {
                "n_contributors": len(post_list),
                "n_works": seqlist_works(post_list),
                "C1": post_c1["C1"],
                "n_used_C1": post_c1["n_used_C1"],
                "n_dropped_alltie_C1": post_c1["n_dropped_alltie_C1"],
                "C2": post_c2,
            },
        },
    }

def pairs_to_lists(pairs):
    logins = sorted(pairs)
    return [pairs[login]["pre"] for login in logins], [pairs[login]["post"] for login in logins]

def cohort_point_stats(pairs, rng):
    pre_list, post_list = pairs_to_lists(pairs)
    return cohort_stats_from_lists(pre_list, post_list, rng)

def sample_pair_lists(pairs, rng):
    logins = sorted(pairs)
    n = len(logins)
    idx = rng.integers(0, n, size=n)
    pre = [pairs[logins[j]]["pre"] for j in idx]
    post = [pairs[logins[j]]["post"] for j in idx]
    return pre, post

def bootstrap_contrast(real_pairs, placebo_pairs, B, rng):
    D = np.empty(B, dtype=float)
    T = np.empty(B, dtype=float)
    for b in range(B):
        real_pre, real_post = sample_pair_lists(real_pairs, rng)
        plac_pre, plac_post = sample_pair_lists(placebo_pairs, rng)
        real = cohort_stats_from_lists(real_pre, real_post, rng)
        plac = cohort_stats_from_lists(plac_pre, plac_post, rng)
        D[b] = real["delta_A"] - plac["delta_A"]
        T[b] = real["delta_TVD"] - plac["delta_TVD"]
    return D, T

def sequence_pairs_from_current_tables(con):
    n_incumbents = int(con.execute("SELECT count(*) FROM incumbents").fetchone()[0])
    seqs_pre, diag_pre = RP.era_sequences_and_diag(con, "pre")
    seqs_post, diag_post = RP.era_sequences_and_diag(con, "post")
    common = sorted(set(seqs_pre) & set(seqs_post))
    pairs = {login: {"pre": seqs_pre[login], "post": seqs_post[login]} for login in common}
    return {
        "n_incumbents": n_incumbents,
        "n_pair_contributors": len(pairs),
        "pairs": pairs,
        "all_realized_diag": {"pre": diag_pre, "post": diag_post},
    }

def add_matched_censor_diag(con, cohort):
    for era in ("pre", "post"):
        row = con.execute(
            f"""
            SELECT count(*) AS n_rows,
                   sum(CASE WHEN b.matched_censored THEN 1 ELSE 0 END) AS n_censored
            FROM base_{era} b
            JOIN incumbents i USING(login)
            WHERE NOT b.extrap_right;
            """
        ).fetchone()
        n_rows = int(row[0] or 0)
        n_censored = int(row[1] or 0)
        cohort["all_realized_diag"][era]["n_rows_before_matched_censor"] = n_rows
        cohort["all_realized_diag"][era]["n_matched_censored"] = n_censored
        cohort["all_realized_diag"][era]["frac_matched_censored"] = (
            n_censored / n_rows if n_rows else float("nan")
        )
    cohort["frac_matched_censored_post"] = cohort["all_realized_diag"]["post"][
        "frac_matched_censored"
    ]

def prepare_placebo_matched_cell_tables(con, parquet_path, half, cell):
    g_expr, incl = RP.credit_sql(cell["credit"])
    c_col = RP.impact_col(cell["proxy"])
    W = int(cell["W"])
    for era, era_where in RP.ERAS["placebo"].items():
        matched_expr = (
            f"({month_add_expr('wr.first_ts', W)} > TIMESTAMP '2020-04-01')"
            if era == "post"
            else "FALSE"
        )
        estimator_keep = "NOT extrap_right AND NOT matched_censored"
        estimator_keep_b = "NOT b.extrap_right AND NOT b.matched_censored"
        con.execute(
            f"""
            CREATE OR REPLACE TEMP TABLE base_{era} AS
            SELECT wr.login, wr.repo, wr.first_ts, wr.dom_eco, wr.work_cohort_year,
                   {g_expr} AS g,
                   ln(1 + GREATEST(wr.{c_col}, 0)) AS lc,
                   wr.extrap_right,
                   {matched_expr} AS matched_censored
            FROM read_parquet('{parquet_path}') wr
            JOIN splt s USING(login)
            JOIN placebo_cohort pc USING(login)
            WHERE wr.W = {W}
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
            WHERE {estimator_keep}
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
            WHERE {estimator_keep_b};
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
            WHERE n >= {int(cell["n_min"])} AND span_mo >= {RP.SPAN_MIN_MO};
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
    for era in RP.ERAS["placebo"]:
        con.execute(
            f"""
            CREATE OR REPLACE TEMP TABLE realized_{era} AS
            SELECT r.*
            FROM realized_{era}_all r
            JOIN incumbents i USING(login);
            """
        )

def prepare_and_extract(con, parquet_path, mode, half, cell, matched_censoring=False):
    if matched_censoring:
        if mode != "placebo":
            raise ValueError("matched_censoring is defined only for the placebo cohort")
        prepare_placebo_matched_cell_tables(con, parquet_path, half, cell)
    else:
        RP.prepare_cell_tables(con, parquet_path, mode, half, cell)
    cohort = sequence_pairs_from_current_tables(con)
    if matched_censoring:
        add_matched_censor_diag(con, cohort)
    return cohort

def build_cell_result(con, parquet_path, half, cell, matched_censoring, seed, B):
    cell_rng = np.random.default_rng(seed)

    real = prepare_and_extract(con, parquet_path, "real", half, cell, matched_censoring=False)
    placebo = prepare_and_extract(
        con, parquet_path, "placebo", half, cell, matched_censoring=matched_censoring
    )

    rec = {
        "cell": dict(cell),
        "credit": cell["credit"],
        "W": cell["W"],
        "n_min": cell["n_min"],
        "proxy": cell["proxy"],
        "primary": is_primary(cell),
        "seed": int(seed),
        "bootstrap_B": int(B),
        "matched_censoring": bool(matched_censoring),
        "cohorts": {
            "real": {
                "n_incumbents": real["n_incumbents"],
                "n_pair_contributors": real["n_pair_contributors"],
                "all_realized_diag": real["all_realized_diag"],
            },
            "placebo": {
                "n_incumbents": placebo["n_incumbents"],
                "n_pair_contributors": placebo["n_pair_contributors"],
                "all_realized_diag": placebo["all_realized_diag"],
            },
        },
    }
    if matched_censoring:
        rec["cohorts"]["placebo"]["frac_matched_censored_post"] = placebo[
            "frac_matched_censored_post"
        ]

    if (
        real["n_pair_contributors"] < RP.MIN_CONTRIBUTORS
        or placebo["n_pair_contributors"] < RP.MIN_CONTRIBUTORS
    ):
        rec["skipped"] = True
        rec["skip_reason"] = (
            "paired contributors below threshold: "
            f"real={real['n_pair_contributors']}, placebo={placebo['n_pair_contributors']}"
        )
        return rec

    real_point = cohort_point_stats(real["pairs"], cell_rng)
    placebo_point = cohort_point_stats(placebo["pairs"], cell_rng)
    D_hat = real_point["delta_A"] - placebo_point["delta_A"]
    T_hat = real_point["delta_TVD"] - placebo_point["delta_TVD"]
    rec["cohorts"]["real"].update(real_point)
    rec["cohorts"]["placebo"].update(placebo_point)
    rec["D_hat"] = float(D_hat)
    rec["T_hat"] = float(T_hat)

    if B > 0:
        draws_D, draws_T = bootstrap_contrast(real["pairs"], placebo["pairs"], B, cell_rng)
        rec["bootstrap"] = {
            "B": int(B),
            "D_ci95": ci(draws_D, 2.5, 97.5),
            "D_ci90": ci(draws_D, 5.0, 95.0),
            "T_ci95": ci(draws_T, 2.5, 97.5),
            "T_ci90": ci(draws_T, 5.0, 95.0),
            "D_p_two_sided_zero": two_sided_zero_p(draws_D),
            "D_TOST_margin": TOST_MARGIN,
            "D_TOST_pass_90ci": bool(
                ci(draws_D, 5.0, 95.0)[0] > -TOST_MARGIN
                and ci(draws_D, 5.0, 95.0)[1] < TOST_MARGIN
            ),
        }
    return rec

def connect_duckdb():
    con = duckdb.connect()
    con.execute("PRAGMA threads=4;")
    tmp = os.environ.get("CODERSQ_DUCKDB_TEMP_DIR")
    if tmp:
        os.makedirs(tmp, exist_ok=True)
        con.execute(f"SET temp_directory='{RP.sql_literal(tmp)}';")
    return con

def output_path(outdir, half, matched_censoring):
    name = f"did_matched_{half}.json" if matched_censoring else f"did_{half}.json"
    return os.path.join(outdir, name)

def run_analysis(
    *,
    half,
    parquet,
    outdir,
    stamp,
    seed=DEFAULT_SEED,
    matched_censoring=False,
    b_primary=1000,
    b_other=200,
    matched_bootstrap_primary_only=True,
):
    if half == "holdout" and os.environ.get("CODERSQ_UNSEAL_HELDOUT") != "1":
        sys.exit(
            "REFUSED: held-out unseal requires CODERSQ_UNSEAL_HELDOUT=1 "
            "(deliberate, timestamped)."
        )

    os.makedirs(outdir, exist_ok=True)
    parquet_path = RP.sql_literal(parquet)
    con = connect_duckdb()
    RP.install_placebo_cohort(con, parquet_path)
    n_total, n_discovery = RP.install_split_table(con, parquet_path)

    grid = RP.make_grid()
    results = []
    final_path = output_path(outdir, half, matched_censoring)
    partial_path = final_path.replace(".json", ".partial.json")

    def payload():
        return {
            "half": half,
            "seed": int(seed),
            "split_salt": RP.SPLIT_SALT,
            "spec_note_path": SPEC_NOTE,
            "timestamp": stamp,
            "matched_censoring": bool(matched_censoring),
            "primary_spec": RP.PRIMARY,
            "bootstrap_B_values": {
                "primary": int(b_primary),
                "other": int(b_other),
                "matched_nonprimary": 0 if matched_bootstrap_primary_only else int(b_other),
            },
            "n_contributors_total": n_total,
            "n_discovery": n_discovery,
            "n_holdout": n_total - n_discovery,
            "results": results,
        }

    log(
        f"contributors total={n_total:,} discovery={n_discovery:,} "
        f"holdout={n_total - n_discovery:,}"
    )
    log(
        f"RUNNING DiD half={half} matched_censoring={matched_censoring} "
        f"parquet={parquet}"
    )

    for i, cell in enumerate(grid):
        primary = is_primary(cell)
        if matched_censoring and matched_bootstrap_primary_only and not primary:
            B = 0
        else:
            B = b_primary if primary else b_other
        seed_i = cell_seed(seed, half, cell, matched_censoring)
        rec = build_cell_result(con, parquet_path, half, cell, matched_censoring, seed_i, B)
        results.append(rec)
        if rec.get("skipped"):
            log(f"[{i + 1}/{len(grid)}] {cell} SKIP: {rec['skip_reason']}")
        else:
            msg = (
                f"[{i + 1}/{len(grid)}] {cell} "
                f"Npair real={rec['cohorts']['real']['n_pair_contributors']:,} "
                f"placebo={rec['cohorts']['placebo']['n_pair_contributors']:,} "
                f"D={rec['D_hat']:.3f} T={rec['T_hat']:.3f} B={B}"
            )
            if matched_censoring:
                msg += (
                    " placebo_post_frac_matched="
                    f"{rec['cohorts']['placebo'].get('frac_matched_censored_post', float('nan')):.3f}"
                )
            log(msg)
        write_json(partial_path, payload())

    write_json(final_path, payload())
    log(f"wrote {final_path} ({len(results)} cells)")
    print("DID_PREPOST_RUN_DONE")
    return final_path, payload()

def add_months(ts, months):
    return pd.Timestamp(ts) + pd.DateOffset(months=int(months))

def discovery_logins(prefix, n):
    out = []
    i = 0
    while len(out) < n:
        login = f"{prefix}_{i:04d}"
        if RP.split_tag(login) == "discovery":
            out.append(login)
        i += 1
    return out

def synthetic_dates(start, offsets):
    return [add_months(start, mo) for mo in offsets]

def append_synthetic_rows(rows, login, cohort, era, dates, lc_values):
    for j, (first_ts, lc) in enumerate(zip(dates, lc_values)):
        c = float(np.exp(lc) - 1.0)
        for W in RP.WS:
            rows.append(
                {
                    "login": login,
                    "repo": f"{login}_{era}_{j:03d}",
                    "first_ts": first_ts,
                    "dom_eco": "synthetic",
                    "work_cohort_year": int(first_ts.year),
                    "credit_share": 1.0,
                    "is_lead": True,
                    "is_core": True,
                    "W": int(W),
                    "c_reach2_sum": c,
                    "c_indeg_sum": c,
                    "extrap_right": False,
                    "synthetic_cohort": cohort,
                }
            )

def build_selftest_parquet(tmpdir):
    rng = np.random.default_rng(424242)
    n = 60
    n_works = 18
    sigma_p = 1.0
    low_q = np.sqrt(0.10)
    high_q = np.sqrt(1.00)
    base = 5.0

    real_logins = discovery_logins("did_real", n)
    placebo_logins = discovery_logins("did_placebo", n)

    real_pre_dates = synthetic_dates("2020-01-01", range(0, 36, 2))[:n_works]
    real_post_dates = synthetic_dates("2023-01-01", range(0, 36, 2))[:n_works]
    placebo_pre_dates = synthetic_dates("2015-01-01", [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 25, 26, 27, 28, 29])
    placebo_post_dates = synthetic_dates("2017-07-01", [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 18, 20, 22, 26, 32])

    rows = []
    for login in real_logins:
        q_pre = rng.normal(0.0, low_q)
        q_post = rng.normal(0.0, high_q)
        pre_lc = base + q_pre + rng.normal(0.0, sigma_p, n_works)
        post_lc = base + q_post + rng.normal(0.0, sigma_p, n_works)
        append_synthetic_rows(rows, login, "real", "pre", real_pre_dates, pre_lc)
        append_synthetic_rows(rows, login, "real", "post", real_post_dates, post_lc)

    for login in placebo_logins:
        q_pre = rng.normal(0.0, low_q)
        q_post = rng.normal(0.0, low_q)
        pre_lc = base + q_pre + rng.normal(0.0, sigma_p, n_works)
        post_lc = base + q_post + rng.normal(0.0, sigma_p, n_works)
        append_synthetic_rows(rows, login, "placebo", "pre", placebo_pre_dates, pre_lc)
        append_synthetic_rows(rows, login, "placebo", "post", placebo_post_dates, post_lc)

    df = pd.DataFrame(rows)
    path = os.path.join(tmpdir, "did_selftest.parquet")
    df.to_parquet(path, index=False)
    return path

def primary_record(payload):
    hits = [r for r in payload["results"] if r.get("primary")]
    if len(hits) != 1:
        raise AssertionError(f"expected exactly one primary record, found {len(hits)}")
    return hits[0]

def run_selftest():
    with tempfile.TemporaryDirectory(prefix="codersq_did_selftest_") as tmp:
        parquet = build_selftest_parquet(tmp)
        outdir = os.path.join(tmp, "out")
        log(f"SELFTEST parquet={parquet}")
        _path, standard = run_analysis(
            half="discovery",
            parquet=parquet,
            outdir=outdir,
            stamp="SELFTEST",
            seed=DEFAULT_SEED,
            matched_censoring=False,
            b_primary=50,
            b_other=50,
            matched_bootstrap_primary_only=False,
        )
        prim = primary_record(standard)
        if prim.get("skipped"):
            raise AssertionError(f"standard primary skipped: {prim.get('skip_reason')}")
        D = prim["D_hat"]
        ci95 = prim["bootstrap"]["D_ci95"]
        if not (0.20 <= D <= 0.65):
            raise AssertionError(f"planted positive D not recovered: D={D}")
        if not (ci95[0] > 0.0 or ci95[1] < 0.0):
            raise AssertionError(f"bootstrap CI did not exclude zero: {ci95}")

        _path_m, matched = run_analysis(
            half="discovery",
            parquet=parquet,
            outdir=outdir,
            stamp="SELFTEST",
            seed=DEFAULT_SEED,
            matched_censoring=True,
            b_primary=50,
            b_other=50,
            matched_bootstrap_primary_only=True,
        )
        prim_m = primary_record(matched)
        frac = prim_m["cohorts"]["placebo"]["frac_matched_censored_post"]
        if not np.isclose(frac, 0.5, atol=1e-12):
            raise AssertionError(f"matched censoring fraction mismatch: {frac}")
        print(
            "SELFTEST PASS "
            f"D_hat={D:.3f} D_ci95=[{ci95[0]:.3f}, {ci95[1]:.3f}] "
            f"matched_frac={frac:.3f}"
        )

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--half", default="discovery", choices=["discovery", "holdout"])
    ap.add_argument("--parquet", default=RP.WORK_REACH)
    ap.add_argument("--outdir", default="/scratch/$USER/codersq/impact/estimates_v2")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--stamp")
    ap.add_argument("--matched-censoring", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    return ap.parse_args()

def main():
    args = parse_args()
    if args.selftest:
        run_selftest()
        return
    if not args.stamp:
        sys.exit("REFUSED: --stamp is required; result JSON timestamps must be caller supplied.")
    run_analysis(
        half=args.half,
        parquet=args.parquet,
        outdir=args.outdir,
        stamp=args.stamp,
        seed=args.seed,
        matched_censoring=args.matched_censoring,
    )

if __name__ == "__main__":
    main()
