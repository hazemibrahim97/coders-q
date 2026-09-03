import json, os
import duckdb

JOIN = os.environ.get("CODERSQ_WORK_REACH", "/scratch/$USER/codersq/impact/joined/work_reach_v2.parquet")
PANEL = os.environ.get("CODERSQ_PANEL", "/scratch/$USER/codersq/impact/panel/career_panel_v2.parquet")
OUT = os.environ.get("CODERSQ_FIG1_OUT", "/scratch/$USER/codersq/impact/estimates_v2/fig1_descriptives.json")

con = duckdb.connect()
con.execute("SET threads=4; SET memory_limit='40GB'; SET temp_directory='/scratch/$USER/codersq/impact/.fig1_spill';")
out = {}

panel_cols = {r[0] for r in con.execute(f"DESCRIBE SELECT * FROM read_parquet('{PANEL}') LIMIT 0").fetchall()}
cohort_expr = "cohort_year" if "cohort_year" in panel_cols else "year(first_ts)"
out["panel_cols"] = sorted(panel_cols)
out["panel_career_len_hist"] = con.execute(f"""
    SELECT least(n, 50) AS n_works, count(*) AS n_logins
    FROM (SELECT login, count(*) AS n FROM read_parquet('{PANEL}') GROUP BY login)
    GROUP BY 1 ORDER BY 1""").fetchall()
out["panel_works_by_cohort"] = con.execute(f"""
    SELECT {cohort_expr} AS cohort_year, count(*) FROM read_parquet('{PANEL}') GROUP BY 1 ORDER BY 1""").fetchall()

out["matched_career_len_hist"] = con.execute(f"""
    SELECT least(n, 50) AS n_works, count(*) AS n_logins
    FROM (SELECT login, count(*) AS n FROM read_parquet('{JOIN}') WHERE W=60 GROUP BY login)
    GROUP BY 1 ORDER BY 1""").fetchall()
out["reach_hist_w60"] = con.execute(f"""
    SELECT CASE WHEN c_reach2_sum <= 0 THEN -1
                ELSE least(floor(log10(c_reach2_sum) * 2) / 2, 6) END AS log10_bin,
           count(*) AS n
    FROM read_parquet('{JOIN}') WHERE W=60 GROUP BY 1 ORDER BY 1""").fetchall()
out["matched_works_by_cohort"] = con.execute(f"""
    SELECT work_cohort_year, count(*) FROM read_parquet('{JOIN}') WHERE W=60 GROUP BY 1 ORDER BY 1""").fetchall()
out["eco_composition_w60"] = con.execute(f"""
    SELECT dom_eco, count(*) AS n_works, count(DISTINCT login) AS n_logins
    FROM read_parquet('{JOIN}') WHERE W=60 GROUP BY 1 ORDER BY n_works DESC LIMIT 15""").fetchall()
out["zero_share_by_W"] = con.execute(f"""
    SELECT W, avg(CASE WHEN c_reach2_sum <= 0 THEN 1.0 ELSE 0.0 END) AS frac_zero, count(*) AS n
    FROM read_parquet('{JOIN}') GROUP BY W ORDER BY W""").fetchall()
join_cols = {r[0] for r in con.execute(f"DESCRIBE SELECT * FROM read_parquet('{JOIN}') LIMIT 0").fetchall()}
hole_expr = "avg(CASE WHEN in_hole THEN 1.0 ELSE 0.0 END)" if "in_hole" in join_cols else "NULL"
out["join_cols"] = sorted(join_cols)
out["censor_flags_by_W"] = con.execute(f"""
    SELECT W, avg(CASE WHEN extrap_right THEN 1.0 ELSE 0.0 END) AS frac_extrap_right,
           {hole_expr} AS frac_in_hole
    FROM read_parquet('{JOIN}') GROUP BY W ORDER BY W""").fetchall()

with open(OUT, "w") as f:
    json.dump(out, f, default=str)
print("wrote", OUT, {k: len(v) for k, v in out.items()})

import shutil
shutil.rmtree("/scratch/$USER/codersq/impact/.fig1_spill", ignore_errors=True)
