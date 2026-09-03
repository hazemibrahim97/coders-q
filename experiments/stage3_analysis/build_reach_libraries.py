import argparse, os, re, subprocess, sys, time
import duckdb

RAW  = "/scratch/$USER/codersq/impact/raw"
WORK = "/scratch/$USER/codersq/impact/work"
OUT  = "/scratch/$USER/codersq/impact/reach"

RUNTIME_KINDS = {'runtime','normal','compile','imports','depends','implementation','api'}

def log(*a): print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)

def snap_date(name):
    m = re.search(r"(\d{4}-\d{2}-\d{2})", name); return m.group(1) if m else "unknown"

def _glob_csv(workdir, kind):
    import glob as _g
    hits = _g.glob(os.path.join(workdir, "**", f"{kind}-*.csv"), recursive=True) + \
           _g.glob(os.path.join(workdir, f"{kind}-*.csv"))
    hits = [h for h in hits if os.path.getsize(h) > 0]
    return hits[0] if hits else None

def extract_both(archive, workdir):
    dep = _glob_csv(workdir, "dependencies")
    prj = _glob_csv(workdir, "projects")
    if dep and prj:
        log(f"reuse extracted CSVs: {os.path.basename(dep)} ({os.path.getsize(dep)/1e9:.1f} GB), "
            f"{os.path.basename(prj)}"); return dep, prj
    if archive.endswith(".zip"):
        names = subprocess.run(["unzip","-Z1",archive], capture_output=True, text=True).stdout.split()
        for kind in ("dependencies","projects"):
            mem = next((n for n in names if re.search(rf"(^|/){kind}-.*\.csv$", n)), None)
            if not mem: raise RuntimeError(f"no {kind}-*.csv in zip: {names[:12]}")
            out = os.path.join(workdir, os.path.basename(mem))
            if not (os.path.exists(out) and os.path.getsize(out) > 0):
                log(f"unzip {mem} ...")
                with open(out, "wb") as w: subprocess.run(["unzip","-p",archive,mem], stdout=w, check=True)
    else:

        log("one-pass tar extract of dependencies-*.csv + projects-*.csv ...")

        subprocess.run(["tar","xzf",archive,"-C",workdir,"--wildcards","--no-anchored",
                        "--exclude=*repository_dependencies*",
                        "dependencies-*.csv","projects-*.csv"], check=True)
    dep = _glob_csv(workdir, "dependencies"); prj = _glob_csv(workdir, "projects")
    if not (dep and prj): raise RuntimeError(f"extraction failed: dep={dep} prj={prj}")
    log(f"  dep={dep} ({os.path.getsize(dep)/1e9:.1f} GB)  prj={prj} ({os.path.getsize(prj)/1e9:.1f} GB)")
    return dep, prj

def norm_repo_sql(col):
    return f"""
      CASE WHEN lower({col}) LIKE '%github.com/%' THEN
        lower(regexp_replace(
          regexp_extract({col}, 'github\\.com[/:]+([^/]+/[^/#?]+)', 1),
          '(\\.git)?/*$', ''))
      ELSE NULL END """

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", required=True)
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--mem", default="56GB"); ap.add_argument("--threads", default="8")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    date = snap_date(args.archive)
    wdir = os.path.join(WORK, f"lib-{date}"); os.makedirs(wdir, exist_ok=True)
    spill = os.path.join(WORK, f"libspill-{date}"); os.makedirs(spill, exist_ok=True)
    arch = os.path.join(RAW, args.archive)

    dep_csv, proj_csv = extract_both(arch, wdir)

    con = duckdb.connect()
    con.execute(f"PRAGMA threads={args.threads}; PRAGMA memory_limit='{args.mem}'; PRAGMA temp_directory='{spill}';")

    con.execute(f"""
        CREATE OR REPLACE TABLE deps AS
        SELECT lower(trim("Platform")) AS child_eco, "Project Name" AS child_name,
               lower(trim("Dependency Platform")) AS parent_eco, "Dependency Name" AS parent_name,
               lower(trim("Dependency Kind")) AS kind, lower(trim("Optional Dependency")) AS optional
        FROM read_csv('{dep_csv}', header=true, all_varchar=true,
                      max_line_size=67108864, ignore_errors=true);
    """)
    nd = con.execute("SELECT count(*) FROM deps").fetchone()[0]
    log(f"deps rows={nd:,}")

    print("\n" + "="*66); print(f"LIBRARIES.IO VALIDATION — {date}"); print("="*66)
    print("[deps] kind distribution (top 20):")
    for r in con.execute("SELECT kind,count(*) c FROM deps GROUP BY kind ORDER BY c DESC LIMIT 20").fetchall():
        print(f"    {r[0]!r:20} {r[1]:,}")
    print("[deps] optional distribution:")
    for r in con.execute("SELECT optional,count(*) c FROM deps GROUP BY optional ORDER BY c DESC LIMIT 6").fetchall():
        print(f"    {r[0]!r:10} {r[1]:,}")

    con.execute("""
        CREATE OR REPLACE TABLE pkg AS
        SELECT eco, name, row_number() OVER () AS pid FROM (
          SELECT DISTINCT child_eco AS eco, child_name AS name FROM deps
          UNION
          SELECT DISTINCT parent_eco AS eco, parent_name AS name FROM deps
        );
    """)
    npk = con.execute("SELECT count(*) FROM pkg").fetchone()[0]
    log(f"distinct packages (nodes)={npk:,}")

    inlist = ", ".join("'" + k + "'" for k in sorted(RUNTIME_KINDS))
    con.execute(f"""
        CREATE OR REPLACE TABLE edges AS
        SELECT DISTINCT cp.pid AS child_id, pp.pid AS parent_id
        FROM deps d
        JOIN pkg cp ON cp.eco=d.child_eco  AND cp.name=d.child_name
        JOIN pkg pp ON pp.eco=d.parent_eco AND pp.name=d.parent_name
        WHERE d.kind IN ({inlist})
          AND COALESCE(d.optional, 'false') IN ('false','f','0','')
          AND cp.pid <> pp.pid;
    """)
    ne = con.execute("SELECT count(*) FROM edges").fetchone()[0]
    log(f"runtime edges={ne:,}")

    con.execute("CREATE OR REPLACE TABLE indeg AS SELECT parent_id, count(DISTINCT child_id) in_degree FROM edges GROUP BY parent_id;")
    ids = con.execute("SELECT count(*),max(in_degree) FROM indeg").fetchone()
    print(f"\n[in-degree] parents>=1 dependent: {ids[0]:,}  max={ids[1]:,}")
    print("[in-degree] top 8 hubs:")
    for r in con.execute("""SELECT p.name,p.eco,i.in_degree FROM indeg i JOIN pkg p ON p.pid=i.parent_id
                            ORDER BY i.in_degree DESC LIMIT 8""").fetchall():
        print(f"    {r[0]!r:26} {r[1]!r:12} {r[2]:,}")

    con.execute(f"""
        CREATE OR REPLACE TABLE proj AS
        SELECT lower(trim("Platform")) AS eco, "Name" AS name, {norm_repo_sql('"Repository URL"')} AS repo_norm
        FROM read_csv('{proj_csv}', header=true, all_varchar=true,
                      max_line_size=67108864, ignore_errors=true);
    """)
    rc = con.execute("SELECT count(*), count(*) FILTER (WHERE repo_norm IS NOT NULL) FROM proj").fetchone()
    print(f"\n[repo bridge] projects={rc[0]:,}  github resolved={rc[1]:,} ({100*rc[1]/max(rc[0],1):.1f}%)")

    if args.validate:
        print("\nLIB_VALIDATE_DONE"); return

    con.execute("""
        CREATE OR REPLACE TABLE pkg_repo AS
        SELECT pk.pid, pk.eco, pk.name, pr.repo_norm
        FROM pkg pk JOIN proj pr ON pr.eco=pk.eco AND pr.name=pk.name
        WHERE pr.repo_norm IS NOT NULL;
    """)
    con.execute("CREATE OR REPLACE TABLE target_parents AS SELECT DISTINCT pid FROM pkg_repo;")
    ntp = con.execute("SELECT count(*) FROM target_parents").fetchone()[0]
    log(f"reach2 for {ntp:,} github package-repo parents ...")
    t0=time.time()
    con.execute("""
        CREATE OR REPLACE TABLE reach2 AS
        WITH e1 AS (SELECT e.parent_id,e.child_id FROM edges e SEMI JOIN target_parents t ON t.pid=e.parent_id),
             h1 AS (SELECT parent_id, child_id AS d FROM e1),
             h2 AS (SELECT e1.parent_id, e2.child_id AS d FROM e1 JOIN edges e2 ON e1.child_id=e2.parent_id),
             alld AS (SELECT parent_id,d FROM h1 UNION ALL SELECT parent_id,d FROM h2)
        SELECT parent_id, count(DISTINCT d) AS reach2 FROM alld WHERE d<>parent_id GROUP BY parent_id;
    """)
    log(f"reach2 done in {(time.time()-t0)/60:.1f} min")

    out_parq = os.path.join(OUT, f"libraries_{date}.parquet")
    con.execute(f"""
        COPY (
          SELECT NULL::BIGINT AS package_id, pr.name AS package, pr.eco AS ecosystem,
                 pr.repo_norm, DATE '{date}' AS snapshot_date, 'libraries' AS source,
                 COALESCE(i.in_degree,0) AS in_degree, COALESCE(r.reach2,0) AS reach2
          FROM pkg_repo pr
          LEFT JOIN indeg  i ON i.parent_id=pr.pid
          LEFT JOIN reach2 r ON r.parent_id=pr.pid
        ) TO '{out_parq}' (FORMAT PARQUET, COMPRESSION ZSTD);
    """)
    st = con.execute(f"""SELECT count(*) AS n_rows, count(DISTINCT repo_norm) AS repos,
                         count(*) FILTER (WHERE reach2>0) AS wr, max(reach2) AS mx
                         FROM read_parquet('{out_parq}')""").fetchone()
    log(f"wrote {out_parq}: {st[0]:,} rows  repos={st[1]:,}  reach2>0={st[2]:,}  max={st[3]:,}")
    print("\nLIB_BUILD_DONE")

if __name__ == "__main__":
    main()
