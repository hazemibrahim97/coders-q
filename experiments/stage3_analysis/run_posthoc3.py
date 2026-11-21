\
\
\
\
\
\
\
\
\
\
\

from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np
import duckdb

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import run_estimators as RE
import qmodel as QM

JOINED = "/scratch/$USER/codersq/impact/joined"
PANELS = {"nohole": "work_reach_v2_nohole.parquet", "clean": "work_reach_v2_clean.parquet"}
VARIANTS = ("base", "nohole", "clean", "founder", "norm_none", "norm_eco", "norm_year", "norm_scale",
            "describe")
NORM_KEYS = {"eco_year": "dom_eco, work_cohort_year", "eco": "dom_eco", "year": "work_cohort_year",
             "scale": "dom_eco, work_cohort_year"}
FOUNDER_BURNIN = "2015-07-01"
N_PERM, N_BOOT_A, N_BOOT_RATIO = 5000, 400, 1000
P = RE.PRIMARY

def log(*a): print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)

def build_sequences_v(con, half, norm="eco_year", founder=False):
\
\

    extra = ("AND EXISTS (SELECT 1 FROM founders f WHERE f.login = t.login AND f.repo = t.repo)"
             if founder else "")
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE cell AS
        SELECT login, repo, first_ts, dom_eco, work_cohort_year,
               credit_share AS g, ln(1 + GREATEST(c_reach2_sum, 0)) AS lc
        FROM read_parquet('{RE.WORK_REACH}') t
        WHERE W = {P['W']} AND (credit_share >= {RE.G_MIN_SHARE:f}) AND NOT extrap_right {extra};
    """)
    logins = [r[0] for r in con.execute("SELECT DISTINCT login FROM cell").fetchall()]
    con.execute("CREATE OR REPLACE TEMP TABLE splt(login VARCHAR, tag VARCHAR);")
    if logins:
        con.executemany("INSERT INTO splt VALUES (?,?)", [(lg, RE.split_tag(lg)) for lg in logins])
    con.execute(f"CREATE OR REPLACE TEMP TABLE cellh AS "
                f"SELECT c.* FROM cell c JOIN splt s USING(login) WHERE s.tag = '{half}';")

    if norm == "none":
        realized = ("SELECT login, repo, first_ts, g * lc AS L, (lc = 0) AS is_zero FROM cellh "
                    "WHERE dom_eco IS NOT NULL")
    else:
        k = NORM_KEYS[norm]

        con.execute(f"CREATE OR REPLACE TEMP TABLE med AS SELECT {k}, median(lc) AS cell_med, "
                    f"coalesce(stddev_samp(lc), 0) AS cell_sd FROM cellh WHERE dom_eco IS NOT NULL GROUP BY {k};")
        Lx = ("CASE WHEN m.cell_sd > 0 THEN h.g * (h.lc - m.cell_med) / m.cell_sd ELSE 0 END"
              if norm == "scale" else "h.g * (h.lc - m.cell_med)")
        realized = (f"SELECT h.login, h.repo, h.first_ts, {Lx} AS L, "
                    f"(h.lc = 0) AS is_zero FROM cellh h JOIN med m USING({k}) WHERE h.dom_eco IS NOT NULL")
    con.execute(f"CREATE OR REPLACE TEMP TABLE realized AS {realized};")
    rows = con.execute(f"""
        WITH agg AS (
          SELECT login, list(L ORDER BY first_ts, repo) AS Ls, count(*) AS n,
                 date_diff('month', min(first_ts), max(first_ts)) AS span_mo,
                 sum(CASE WHEN is_zero THEN 1 ELSE 0 END) AS n_zero
          FROM realized GROUP BY login)
        SELECT login, Ls, n, n_zero FROM agg WHERE n >= {P['n_min']} AND span_mo >= {RE.SPAN_MIN_MO};
    """).fetchall()
    seqs = {lg: np.exp(np.asarray(Ls, dtype=float)) for lg, Ls, _, _ in rows}
    nworks = sum(r[2] for r in rows); nz = sum(r[3] for r in rows)
    return seqs, dict(n_contributors=len(seqs), n_works=nworks,
                      frac_zero=(nz / nworks if nworks else float("nan")))

def founder_table(con):
\
\
\

    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE works AS
        SELECT DISTINCT login, repo, repo_lower, first_ts FROM read_parquet('{RE.WORK_REACH}');
        CREATE OR REPLACE TEMP TABLE founders AS
        SELECT w.login, w.repo FROM works w
        JOIN (SELECT repo_lower, min(first_ts) AS t0 FROM works GROUP BY repo_lower) r USING(repo_lower)
        WHERE w.first_ts = r.t0 AND r.t0 >= TIMESTAMPTZ '{FOUNDER_BURNIN} 00:00:00+00';
    """)
    n_all, n_f = con.execute("SELECT (SELECT count(*) FROM works), (SELECT count(*) FROM founders)").fetchone()
    return dict(n_works_panel=n_all, n_founder_works_panel=n_f, founder_share_panel=n_f / n_all)

def ratio_bootstrap(seqs, rng):

    seqlist = list(seqs.values()); n = len(seqlist)
    rs = np.array([QM.decompose_autocorr([seqlist[j] for j in rng.integers(0, n, size=n)],
                                         max_lag=8, n_boot=0)["persistent_stableQ_ratio"]
                   for _ in range(N_BOOT_RATIO)])
    return {"n_draws": N_BOOT_RATIO, "ci95": [float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5))],
            "share_gt_0p20": float(np.mean(rs > 0.20)), "median": float(np.median(rs))}

def describe(con, seqs):

    con.execute("CREATE OR REPLACE TEMP TABLE inc(login VARCHAR);")
    con.executemany("INSERT INTO inc VALUES (?)", [(lg,) for lg in seqs])
    con.execute("""CREATE OR REPLACE TEMP TABLE dw AS
        SELECT h.* FROM realized r JOIN inc USING(login)
        JOIN cellh h ON h.login = r.login AND h.repo = r.repo AND h.first_ts = r.first_ts;""")
    q = lambda s: con.execute(s).fetchone()
    car = q("""SELECT count(*), sum(n),
                 quantile_cont(n, [0.25,0.5,0.75]), quantile_cont(span, [0.25,0.5,0.75])
               FROM (SELECT login, count(*) n, date_diff('month', min(first_ts), max(first_ts)) span
                     FROM dw GROUP BY login)""")
    bins = con.execute("""SELECT CASE WHEN y <= 2016 THEN '2015-16' WHEN y <= 2019 THEN '2017-19'
                                      WHEN y <= 2022 THEN '2020-22' ELSE '2023-25' END b, count(*)
                          FROM (SELECT login, year(min(first_ts) AT TIME ZONE 'UTC') y FROM dw GROUP BY login)
                          GROUP BY b ORDER BY b""").fetchall()
    zero, g = q("SELECT avg(CASE WHEN lc = 0 THEN 1.0 ELSE 0.0 END), quantile_cont(g, [0.25,0.5,0.75]) FROM dw")
    eco = con.execute("SELECT coalesce(dom_eco, '(none)'), count(*) FROM dw GROUP BY 1 ORDER BY 2 DESC").fetchall()
    nw = car[1]
    top = [[e, c / nw] for e, c in eco[:5]]
    iqr = lambda v: {"median": v[1], "p25": v[0], "p75": v[2]}
    return {"n_careers": car[0], "n_works": nw, "works_per_career": iqr(car[2]),
            "span_months": iqr(car[3]),
            "first_work_year_share": {b: c / car[0] for b, c in bins},
            "share_works_zero_reach": zero, "credit_share": iqr(g),
            "top_ecosystems_share_of_works": top + [["other", sum(c for _, c in eco[5:]) / nw]]}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=VARIANTS)
    ap.add_argument("--half", required=True, choices=["discovery", "holdout"])
    ap.add_argument("--out", default="/scratch/$USER/codersq/impact/estimates_v2_posthoc3")
    ap.add_argument("--seed", type=int, default=20261121)
    a = ap.parse_args()
    if a.half == "holdout" and os.environ.get("CODERSQ_UNSEAL_HELDOUT") != "1":
        sys.exit("REFUSED: held-out unseal requires CODERSQ_UNSEAL_HELDOUT=1.")
    RE.WORK_REACH = os.path.join(JOINED, PANELS.get(a.variant, "work_reach_v2.parquet"))
    os.makedirs(a.out, exist_ok=True)
    con = duckdb.connect(); con.execute("PRAGMA threads=4;")
    tmp = os.environ.get("CODERSQ_DUCKDB_TEMP_DIR")
    if tmp: con.execute(f"SET temp_directory='{tmp}';")
    rng = np.random.default_rng(a.seed)
    log(f"variant={a.variant} half={a.half} panel={RE.WORK_REACH} seed={a.seed}")
    out = {"analysis": "posthoc3", "variant": a.variant, "half": a.half, "seed": a.seed,
           "spec_note_path": "prereg/POSTHOC-NOTE-3-camera-ready-robustness.md",
           "label": "post-unsealing exploratory (post-hoc)", "panel_path": RE.WORK_REACH,
           "primary_cell": P, "split_salt": RE.SPLIT_SALT}

    if a.variant in ("base", "nohole", "clean", "describe"):
        seqs, diag = RE.build_sequences(con, a.half, P["credit"], P["W"], P["n_min"], P["proxy"])
        if a.variant == "base":
            s2, d2 = build_sequences_v(con, a.half)
            assert d2 == diag and s2.keys() == seqs.keys() and all(
                np.array_equal(seqs[k], s2[k]) for k in seqs), "copied builder != frozen builder"
            log("copied builder == frozen builder (exact)")

            out["median_zero"] = dict(zip(
                ("n_cells", "share_cells_med0", "n_cell_works", "share_cell_works_med0"),
                con.execute("""SELECT (SELECT count(*) FROM med), (SELECT avg((cell_med = 0)::INT) FROM med),
                    count(*), avg((m.cell_med = 0)::INT) FROM cellh h
                    JOIN med m USING(dom_eco, work_cohort_year)""").fetchone()))
        if a.variant == "describe":
            out["describe"] = describe(con, seqs)
    elif a.variant == "founder":
        out["founder_panel"] = founder_table(con)
        build_sequences_v(con, a.half)
        n_cell, n_cell_f = con.execute(
            "SELECT count(*), count(*) FILTER (WHERE EXISTS (SELECT 1 FROM founders f "
            "WHERE f.login = h.login AND f.repo = h.repo)) FROM cellh h").fetchone()
        out["founder_cell"] = {"n_works_cell_half": n_cell, "n_founder_works_cell_half": n_cell_f,
                               "founder_share_cell_half": n_cell_f / n_cell}
        seqs, diag = build_sequences_v(con, a.half, founder=True)
    else:
        seqs, diag = build_sequences_v(con, a.half, norm=a.variant.split("_")[1])
    out.update(diag)
    log(f"N={diag['n_contributors']:,} works={diag['n_works']:,}")

    if a.variant != "describe" and diag["n_contributors"] >= 30:
        r = RE.run_cell(seqs, rng, N_PERM, N_BOOT_A)
        out.update(r)
        out["ratio_boot"] = ratio_bootstrap(seqs, rng)
        c1, c2 = r["C1"], r["C2"]
        log(f"C1 meanP={c1['mean_P']:.3f} null={c1['null_mean_P']:.3f} tvd={c1['tvd']:.3f} p={c1['perm_p']} "
            f"| C2 A={c2['A_stableQ_share']:.3f} {c2['A_ci']} B={c2['B_AR_share']:.3f} rho={c2['rho']:.2f} "
            f"ratio={c2['ratio_A_over_AB']:.3f} {out['ratio_boot']}")
    elif a.variant != "describe":
        out["skipped"] = "fewer than 30 contributors"

    path = os.path.join(a.out, f"posthoc3_{a.variant}_{a.half}.json")
    with open(path, "w") as f: json.dump(out, f, indent=2, default=float)
    log(f"wrote {path}")
    print("POSTHOC3_DONE")

if __name__ == "__main__":
    main()
