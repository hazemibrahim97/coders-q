import argparse, glob, os, time
import duckdb

BASE  = "/scratch/$USER/codersq/impact"

PANEL = os.environ.get("CODERSQ_PANEL", f"{BASE}/panel/career_panel.parquet")
REACH = f"{BASE}/reach"
OUTD  = f"{BASE}/joined"
JOIN_OUT = os.environ.get("CODERSQ_JOIN_OUT", f"{OUTD}/work_reach.parquet")
WS = (24, 36, 60, 120)
HOLE_L, HOLE_R = "2020-01-12", "2022-08-17"
N_SNAPSHOTS_FULL = int(os.environ.get("CODERSQ_N_SNAPSHOTS", "12"))

def log(*a): print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--mem", default="56GB"); ap.add_argument("--threads", default="8")
    args = ap.parse_args()

    parqs = sorted(glob.glob(f"{REACH}/*.parquet"))
    log(f"reach parquets found: {len(parqs)}")
    for p in parqs: print("   ", os.path.basename(p), flush=True)
    if not args.validate and len(parqs) < N_SNAPSHOTS_FULL:
        raise SystemExit(f"FULL BUILD REFUSED: {len(parqs)}/{N_SNAPSHOTS_FULL} snapshots present")

    os.makedirs(OUTD, exist_ok=True)
    spill = f"{BASE}/work/joinspill"; os.makedirs(spill, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"PRAGMA threads={args.threads};")
    con.execute(f"PRAGMA memory_limit='{args.mem}';")
    con.execute(f"PRAGMA temp_directory='{spill}';")

    con.execute(f"""CREATE VIEW reach AS
        SELECT * FROM read_parquet({parqs!r}, union_by_name=true)
        WHERE repo_norm IS NOT NULL;""")
    con.execute(f"""CREATE VIEW panel AS
        SELECT login, repo, lower(repo) AS repo_lower, first_ts, last_ts, commits_by_login,
               repo_commits, credit_share, is_lead, is_core, country, work_cohort_year
        FROM read_parquet('{PANEL}');""")
    n_panel, n_repos = con.execute(
        "SELECT count(*), count(DISTINCT repo_lower) FROM panel").fetchone()
    log(f"panel: {n_panel:,} works, {n_repos:,} distinct repos")

    cal = con.execute(
        "SELECT DISTINCT snapshot_date, source FROM reach ORDER BY snapshot_date").fetchall()
    print("[calendar]", *[f"{d} ({s})" for d, s in cal], sep="\n    ", flush=True)
    snap_dates = [d for d, _ in cal]
    first_snap, last_snap = snap_dates[0], snap_dates[-1]

    con.execute("""
        CREATE OR REPLACE TABLE repo_snap AS
        SELECT repo_norm, snapshot_date,
               any_value(source)          AS source,
               SUM(reach2)                AS reach2_sum,
               MAX(reach2)                AS reach2_max,
               SUM(in_degree)             AS indeg_sum,
               MAX(in_degree)             AS indeg_max,
               count(*)                   AS n_pkgs,
               arg_max(ecosystem, reach2) AS dom_eco
        FROM reach GROUP BY repo_norm, snapshot_date;""")

    con.execute("""CREATE OR REPLACE TABLE matched_repos AS
        SELECT repo_norm, count(DISTINCT snapshot_date) AS n_snaps,
               min(snapshot_date) AS first_seen, max(snapshot_date) AS last_seen
        FROM repo_snap GROUP BY repo_norm;""")

    print("\n" + "=" * 70 + f"\nPANEL x REACH COVERAGE — {len(parqs)} snapshots\n" + "=" * 70)
    mr = con.execute("""
        SELECT count(DISTINCT p.repo_lower)                                       AS panel_repos,
               count(DISTINCT p.repo_lower) FILTER (WHERE m.repo_norm IS NOT NULL) AS matched_repos,
               count(*)                                                           AS works,
               count(*)              FILTER (WHERE m.repo_norm IS NOT NULL)        AS matched_works
        FROM panel p LEFT JOIN matched_repos m ON m.repo_norm = p.repo_lower""").fetchone()
    print(f"[match] repos: {mr[1]:,}/{mr[0]:,} matched ({100*mr[1]/mr[0]:.2f}%)   "
          f"works: {mr[3]:,}/{mr[2]:,} matched ({100*mr[3]/mr[2]:.2f}%)")
    print("[match] matched works by cohort year:")
    for r in con.execute("""
        SELECT p.work_cohort_year, count(*) w,
               count(*) FILTER (WHERE m.repo_norm IS NOT NULL) mw
        FROM panel p LEFT JOIN matched_repos m ON m.repo_norm=p.repo_lower
        GROUP BY 1 ORDER BY 1""").fetchall():
        print(f"    {r[0]}  {r[2]:,}/{r[1]:,} ({100*r[2]/r[1]:.2f}%)")
    print("[match] matched repos by dominant ecosystem (latest snapshot), top 15:")
    for r in con.execute("""
        WITH latest AS (SELECT repo_norm, arg_max(dom_eco, snapshot_date) AS eco
                        FROM repo_snap GROUP BY repo_norm)
        SELECT l.eco, count(*) FROM latest l
        JOIN (SELECT DISTINCT repo_lower FROM panel) p ON p.repo_lower=l.repo_norm
        GROUP BY 1 ORDER BY 2 DESC LIMIT 15""").fetchall():
        print(f"    {r[0]!r:16} {r[1]:,}")

    ws_sql = ", ".join(f"({w})" for w in WS)
    con.execute(f"""
        CREATE OR REPLACE TABLE work_h AS
        SELECT p.*, w.W,
               CAST(p.first_ts + to_months(w.W) AS DATE) AS h
        FROM panel p
        CROSS JOIN (VALUES {ws_sql}) AS w(W)
        SEMI JOIN matched_repos m ON m.repo_norm = p.repo_lower;""")
    con.execute("""CREATE OR REPLACE TABLE repo_h AS
        SELECT DISTINCT repo_lower, W, h FROM work_h;""")
    n_rh = con.execute("SELECT count(*) FROM repo_h").fetchone()[0]
    log(f"distinct (repo, W, horizon) tuples: {n_rh:,}")

    dates_sql = ", ".join(f"(DATE '{d}')" for d in snap_dates)
    con.execute(f"CREATE OR REPLACE TABLE cal AS SELECT * FROM (VALUES {dates_sql}) t(snap);")
    con.execute("""CREATE OR REPLACE TABLE cal_iv AS
        SELECT snap AS lo, LEAD(snap) OVER (ORDER BY snap) AS hi FROM cal;""")
    con.execute(f"""
        CREATE OR REPLACE TABLE repo_h_br AS
        SELECT r.repo_lower, r.W, r.h,
               iv.lo AS snap_prev,
               CASE WHEN r.h = iv.lo THEN iv.lo
                    WHEN iv.lo IS NULL THEN DATE '{first_snap}'
                    ELSE iv.hi END AS snap_next
        FROM repo_h r
        LEFT JOIN cal_iv iv ON r.h >= iv.lo AND (iv.hi IS NULL OR r.h < iv.hi);""")

    con.execute(f"""
        CREATE OR REPLACE TABLE repo_h_val AS
        SELECT b.*,
               COALESCE(sp.reach2_sum,0) AS v_prev_sum, COALESCE(sn.reach2_sum,0) AS v_next_sum,
               COALESCE(sp.reach2_max,0) AS v_prev_max, COALESCE(sn.reach2_max,0) AS v_next_max,
               COALESCE(sp.indeg_sum,0)  AS d_prev_sum, COALESCE(sn.indeg_sum,0)  AS d_next_sum,
               COALESCE(sn.n_pkgs, sp.n_pkgs)   AS n_pkgs,
               COALESCE(sn.dom_eco, sp.dom_eco) AS dom_eco
        FROM repo_h_br b
        LEFT JOIN repo_snap sp ON sp.repo_norm=b.repo_lower AND sp.snapshot_date=b.snap_prev
        LEFT JOIN repo_snap sn ON sn.repo_norm=b.repo_lower AND sn.snapshot_date=b.snap_next;""")

    interp = """
        CASE
          WHEN snap_prev IS NULL THEN v_next_{m}                                  -- extrap_left
          WHEN snap_next IS NULL THEN v_prev_{m}                                  -- extrap_right
          WHEN snap_prev = snap_next THEN v_prev_{m}                              -- exact hit
          ELSE exp( ln(1+v_prev_{m}) + (ln(1+v_next_{m}) - ln(1+v_prev_{m}))
                    * CAST(h - snap_prev AS DOUBLE) / CAST(snap_next - snap_prev AS DOUBLE) ) - 1
        END"""
    con.execute(f"""
        CREATE OR REPLACE TABLE repo_h_c AS
        SELECT repo_lower, W, h, snap_prev, snap_next, n_pkgs, dom_eco,
               {interp.format(m='sum')} AS c_reach2_sum,
               {interp.format(m='max')} AS c_reach2_max,
               {interp.format(m='sum').replace('v_prev_sum','d_prev_sum').replace('v_next_sum','d_next_sum')}
                                        AS c_indeg_sum,
               (snap_prev IS NULL)                             AS extrap_left,
               (snap_next IS NULL)                             AS extrap_right,
               (snap_prev = DATE '{HOLE_L}' AND snap_next = DATE '{HOLE_R}') AS in_hole,
               LEAST(COALESCE(h - snap_prev, 99999), COALESCE(snap_next - h, 99999)) AS gap_days
        FROM repo_h_val;""")

    print("\n[horizon] per W: extrapolation / hole / gap profile (matched repo-horizons):")
    for r in con.execute("""
        SELECT W, count(*) n,
               count(*) FILTER (WHERE extrap_left)  el,
               count(*) FILTER (WHERE extrap_right) er,
               count(*) FILTER (WHERE in_hole)      ih,
               median(gap_days) med_gap,
               quantile_cont(gap_days, 0.9) p90_gap
        FROM repo_h_c GROUP BY W ORDER BY W""").fetchall():
        print(f"    W={r[0]:>3}: n={r[1]:,}  extrapL={100*r[2]/r[1]:.1f}%  extrapR={100*r[3]/r[1]:.1f}%"
              f"  in_hole={100*r[4]/r[1]:.1f}%  gap_days med={r[5]:.0f} p90={r[6]:.0f}")
    print("[horizon] c_reach2_sum quantiles at W=60 (matched repo-horizons):")
    q = con.execute("""SELECT count(*) , count(*) FILTER (WHERE c_reach2_sum=0),
               quantile_cont(c_reach2_sum,[0.5,0.9,0.99,0.999]) FROM repo_h_c WHERE W=60""").fetchone()
    print(f"    n={q[0]:,}  zeros={100*q[1]/q[0]:.1f}%  p50/p90/p99/p99.9={[round(x,1) for x in q[2]]}")

    if args.validate:

        print("\n[sanity] lodash/lodash + facebook/react horizon values (any W):")
        for r in con.execute("""
            SELECT repo_lower, W, h, snap_prev, snap_next, round(c_reach2_sum,1), in_hole
            FROM repo_h_c WHERE repo_lower IN ('lodash/lodash','facebook/react')
            ORDER BY repo_lower, W LIMIT 16""").fetchall():
            print("   ", r)
        print("\nVALIDATE_REPORT_DONE")
        return

    out = JOIN_OUT
    con.execute(f"""
        COPY (
          SELECT w.login, w.repo, w.repo_lower, w.first_ts, w.last_ts, w.commits_by_login,
                 w.repo_commits, w.credit_share, w.is_lead, w.is_core, w.country,
                 w.work_cohort_year, w.W, w.h,
                 c.snap_prev, c.snap_next, c.n_pkgs, c.dom_eco,
                 c.c_reach2_sum, c.c_reach2_max, c.c_indeg_sum,
                 c.extrap_left, c.extrap_right, c.in_hole, c.gap_days,
                 (w.W = 60 AND w.first_ts >= TIMESTAMP '2017-08-01') AS clean_cadence
          FROM work_h w
          JOIN repo_h_c c ON c.repo_lower=w.repo_lower AND c.W=w.W AND c.h=w.h
        ) TO '{out}' (FORMAT PARQUET, COMPRESSION ZSTD);""")
    st = con.execute(f"""SELECT count(*), count(DISTINCT login), count(DISTINCT repo_lower),
               count(*) FILTER (WHERE W=60) FROM read_parquet('{out}')""").fetchone()
    log(f"wrote {out}: {st[0]:,} work-horizon rows  logins={st[1]:,}  repos={st[2]:,}  W60 rows={st[3]:,}")
    print("\nJOIN_BUILD_DONE")

if __name__ == "__main__":
    main()
