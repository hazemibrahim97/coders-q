import argparse, os, re, subprocess, sys, time, glob
import duckdb

SIF  = "/scratch/$USER/codersq/impact/postgres14.sif"
RAW  = "/scratch/$USER/codersq/impact/raw"
WORK = "/scratch/$USER/codersq/impact/work"
OUT  = "/scratch/$USER/codersq/impact/reach"
LOGS = "/scratch/$USER/codersq/impact/logs"

def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)

def snapshot_date(tarname):
    m = re.search(r"(\d{4}-\d{2}-\d{2})", tarname)
    return m.group(1) if m else "unknown"

def extract_sql(tarname):
    date = snapshot_date(tarname)
    member = f"packages-{date}/packages-{date}.sql"
    sqlp = os.path.join(WORK, member)
    if not (os.path.exists(sqlp) and os.path.getsize(sqlp) > 0):
        log(f"extracting {member} ...")
        subprocess.run(["tar","xzf",os.path.join(RAW,tarname),"-C",WORK,member], check=True)
    log(f"sql = {sqlp} ({os.path.getsize(sqlp)/1e9:.1f} GB)")
    return sqlp

COPY_RE_TMPL = r"^COPY public\.{t} \((.*?)\) FROM stdin;"

def header_cols(sqlp, table):
    cmd = (f"singularity exec -B {WORK} {SIF} pg_restore --data-only -t {table} -f - {sqlp} "
           f"2>/dev/null | grep -m1 '^COPY public.{table} '")
    out = subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout
    m = re.search(r"\((.*?)\) FROM stdin;", out)
    if not m:
        raise RuntimeError(f"no COPY header for {table}")
    return [c.strip().strip('"') for c in m.group(1).split(",")]

def restore_table(sqlp, table, out_tsv, reuse=True):
    if reuse and os.path.exists(out_tsv) and os.path.getsize(out_tsv) > 0:
        cols = header_cols(sqlp, table)
        log(f"reuse {table}: existing tsv {os.path.getsize(out_tsv)/1e9:.1f} GB, cols={cols}")
        return cols, -1
    cmd = ["singularity","exec","-B",WORK,SIF,"pg_restore","--data-only","-t",table,"-f","-",sqlp]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                            bufsize=1<<22, text=True)
    copy_re = re.compile(COPY_RE_TMPL.format(t=re.escape(table)))
    cols, in_data, n = None, False, 0
    with open(out_tsv, "w") as w:
        for line in proc.stdout:
            if not in_data:
                m = copy_re.match(line)
                if m:
                    cols = [c.strip().strip('"') for c in m.group(1).split(",")]
                    in_data = True
                continue
            if line.startswith("\\."):
                break
            w.write(line); n += 1
    proc.stdout.close(); proc.wait()
    if cols is None:
        raise RuntimeError(f"no COPY header found for {table}")
    log(f"restored {table}: {n:,} rows, cols={cols}")
    return cols, n

KEEP = {
    "packages":     ["id", "registry_id", "name", "ecosystem", "repository_url"],
    "versions":     ["id", "package_id"],
    "dependencies": ["version_id", "package_name", "ecosystem", "kind", "optional"],
    "registries":   ["id", "name", "ecosystem"],
}

def load_tsv(con, name, tsv, cols):
    colspec = ", ".join([f"'{c}': 'VARCHAR'" for c in cols])
    keep = [c for c in KEEP.get(name, cols) if c in cols]
    con.execute(f"""
        CREATE OR REPLACE TABLE {name} AS
        SELECT {', '.join(keep)} FROM read_csv('{tsv}', delim='\t', header=false, quote='', escape='',
               nullstr='\\N', columns={{{colspec}}}, auto_detect=false, ignore_errors=false,
               max_line_size=67108864);
    """)
    n = con.execute(f"SELECT count(*) FROM {name}").fetchone()[0]
    log(f"loaded {name}: {n:,} rows into duckdb (kept cols: {keep})")
    return n

def norm_repo_sql(col):
    return f"""
      CASE WHEN lower({col}) LIKE '%github.com/%' THEN
        lower(regexp_replace(
          regexp_extract({col}, 'github\\.com[/:]+([^/]+/[^/#?]+)', 1),
          '(\\.git)?/*$', ''))
      ELSE NULL END
    """

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True, help="tar filename in RAW")
    ap.add_argument("--validate", action="store_true", help="Phase-1 validation only")
    ap.add_argument("--mem", default="48GB")
    ap.add_argument("--threads", default="8")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    date = snapshot_date(args.snapshot)
    wdir = os.path.join(WORK, f"tsv-{date}"); os.makedirs(wdir, exist_ok=True)
    spill = os.path.join(WORK, f"spill-{date}"); os.makedirs(spill, exist_ok=True)

    sqlp = extract_sql(args.snapshot)

    tsvs = {}
    schema = {}
    for t in ("packages","versions","dependencies","registries"):
        tsv = os.path.join(wdir, f"{t}.tsv")
        cols, n = restore_table(sqlp, t, tsv)
        tsvs[t] = tsv; schema[t] = (cols, n)

    con = duckdb.connect()
    con.execute(f"PRAGMA threads={args.threads};")
    con.execute(f"PRAGMA memory_limit='{args.mem}';")
    con.execute(f"PRAGMA temp_directory='{spill}';")

    for t in ("packages","versions","dependencies","registries"):
        load_tsv(con, t, tsvs[t], schema[t][0])

    print("\n" + "="*70)
    print(f"VALIDATION REPORT — snapshot {date}")
    print("="*70)

    print("\n[versions] columns:", schema["versions"][0])
    vnull = con.execute("""
        SELECT count(*) tot,
               count(*) FILTER (WHERE package_id IS NULL) pkgid_null,
               count(*) FILTER (WHERE TRY_CAST(package_id AS BIGINT) IS NOT NULL) pkgid_ok
        FROM versions""").fetchone()
    print(f"[versions] rows={vnull[0]:,}  package_id NULL={vnull[1]:,}  castable={vnull[2]:,}")

    dnull = con.execute("""
        SELECT count(*) tot,
               count(*) FILTER (WHERE version_id IS NULL) verid_null,
               count(DISTINCT kind) nkind
        FROM dependencies""").fetchone()
    print(f"[dependencies] rows={dnull[0]:,}  version_id NULL={dnull[1]:,}  distinct kinds={dnull[2]}")
    print("[dependencies] FULL kind distribution (all values):")
    for r in con.execute("SELECT kind, count(*) c FROM dependencies GROUP BY kind ORDER BY c DESC").fetchall():
        print(f"    {r[0]!r:22} {r[1]:,}")
    print("[dependencies] optional distribution:")
    for r in con.execute("SELECT optional, count(*) c FROM dependencies GROUP BY optional ORDER BY c DESC").fetchall():
        print(f"    {r[0]!r:8} {r[1]:,}")
    print("[dependencies] kind x ecosystem (top 30 (ecosystem,kind) cells) — to build runtime mapping:")
    for r in con.execute("""
        SELECT ecosystem, kind, count(*) c FROM dependencies
        GROUP BY ecosystem, kind ORDER BY c DESC LIMIT 30""").fetchall():
        print(f"    {r[0]!r:14} {r[1]!r:16} {r[2]:,}")

    con.execute("""
        CREATE OR REPLACE TABLE dep_child AS
        SELECT TRY_CAST(v.package_id AS BIGINT) AS child_id,
               d.package_name AS parent_name, lower(trim(d.ecosystem)) AS eco,
               lower(trim(d.kind)) AS kind, d.optional
        FROM dependencies d
        LEFT JOIN versions v ON TRY_CAST(d.version_id AS BIGINT)=TRY_CAST(v.id AS BIGINT);
    """)
    jc = con.execute("""
        SELECT count(*) tot,
               count(*) FILTER (WHERE child_id IS NOT NULL) n_match
        FROM dep_child""").fetchone()
    print(f"\n[join] dep -> versions -> child_id: {jc[1]:,}/{jc[0]:,} matched ({100*jc[1]/max(jc[0],1):.1f}%)")

    con.execute("""
        CREATE OR REPLACE TABLE pkg_by_name AS
        SELECT lower(trim(ecosystem)) AS eco, name,
               min(TRY_CAST(id AS BIGINT)) AS pkg_id, count(*) AS n_ids
        FROM packages GROUP BY lower(trim(ecosystem)), name;
    """)
    amb = con.execute("SELECT count(*) FILTER (WHERE n_ids>1), count(*) FROM pkg_by_name").fetchone()
    print(f"[parent] (ecosystem,name) groups: {amb[1]:,}  ambiguous(>1 id)={amb[0]:,} ({100*amb[0]/max(amb[1],1):.2f}%)")
    jp = con.execute("""
        SELECT count(*) tot, count(*) FILTER (WHERE p.pkg_id IS NOT NULL) n_match
        FROM dep_child dc LEFT JOIN pkg_by_name p ON p.eco=dc.eco AND p.name=dc.parent_name
    """).fetchone()
    print(f"[join] parent_name -> packages.id: {jp[1]:,}/{jp[0]:,} matched ({100*jp[1]/max(jp[0],1):.1f}%)")

    RUNTIME_KINDS = {'runtime','normal','compile','imports','depends','implementation','api'}
    inlist = ", ".join("'" + k.replace("'", "''") + "'" for k in sorted(RUNTIME_KINDS))
    con.execute(f"""
        CREATE OR REPLACE TABLE edges AS
        SELECT DISTINCT dc.child_id, p.pkg_id AS parent_id
        FROM dep_child dc
        JOIN pkg_by_name p ON p.eco=dc.eco AND p.name=dc.parent_name
        WHERE dc.kind IN ({inlist}) AND dc.optional='f'
          AND dc.child_id IS NOT NULL AND p.pkg_id IS NOT NULL
          AND dc.child_id <> p.pkg_id;
    """)
    ne = con.execute("SELECT count(*) FROM edges").fetchone()[0]
    ne_strict = con.execute("""
        SELECT count(*) FROM (
          SELECT DISTINCT dc.child_id, p.pkg_id AS parent_id
          FROM dep_child dc JOIN pkg_by_name p ON p.eco=dc.eco AND p.name=dc.parent_name
          WHERE dc.kind='runtime' AND dc.optional='f'
            AND dc.child_id IS NOT NULL AND p.pkg_id IS NOT NULL AND dc.child_id <> p.pkg_id)
    """).fetchone()[0]
    print(f"\n[edges] distinct child->parent edges: BROAD runtime-set={ne:,}  vs  STRICT kind='runtime'={ne_strict:,}")

    con.execute("""
        CREATE OR REPLACE TABLE indeg AS
        SELECT parent_id, count(DISTINCT child_id) AS in_degree
        FROM edges GROUP BY parent_id;
    """)
    id_stats = con.execute("SELECT count(*), max(in_degree), avg(in_degree) FROM indeg").fetchone()
    print(f"[in-degree] parents with >=1 dependent: {id_stats[0]:,}  max={id_stats[1]:,}  mean={id_stats[2]:.2f}")
    print("[in-degree] top 10 hubs (by direct dependents):")
    for r in con.execute("""
        SELECT pk.name, pk.ecosystem, i.in_degree
        FROM indeg i JOIN packages pk ON TRY_CAST(pk.id AS BIGINT)=i.parent_id
        ORDER BY i.in_degree DESC LIMIT 10""").fetchall():
        print(f"    {r[0]!r:28} {r[1]!r:14} {r[2]:,}")

    con.execute(f"""
        CREATE OR REPLACE TABLE pkg_repo AS
        SELECT TRY_CAST(id AS BIGINT) AS pkg_id, name, ecosystem,
               {norm_repo_sql('repository_url')} AS repo_norm
        FROM packages;
    """)
    rc = con.execute("""
        SELECT count(*) tot,
               count(*) FILTER (WHERE repo_norm IS NOT NULL) gh,
               count(DISTINCT repo_norm) FILTER (WHERE repo_norm IS NOT NULL) distinct_repos
        FROM pkg_repo""").fetchone()
    print(f"\n[repo bridge] packages={rc[0]:,}  github repo_norm resolved={rc[1]:,} ({100*rc[1]/max(rc[0],1):.1f}%)  distinct repos={rc[2]:,}")
    print("[repo bridge] sample repo_norm:")
    for r in con.execute("SELECT name, ecosystem, repo_norm FROM pkg_repo WHERE repo_norm IS NOT NULL LIMIT 6").fetchall():
        print(f"    {r[0]!r:24} {r[1]!r:12} {r[2]}")

    print("\n[reach2] probing exact 2-hop distinct-descendant cost on a sample of 2,000 parents...")
    con.execute("""
        CREATE OR REPLACE TABLE sample_parents AS
        (SELECT parent_id FROM indeg ORDER BY in_degree DESC LIMIT 1000)
        UNION
        (SELECT parent_id FROM indeg USING SAMPLE 1000 ROWS);
    """)
    t0 = time.time()
    con.execute("""
        CREATE OR REPLACE TABLE reach2_sample AS
        WITH sp AS (SELECT parent_id FROM sample_parents),
             e1 AS (SELECT e.parent_id, e.child_id FROM edges e JOIN sp ON sp.parent_id=e.parent_id),
             h1 AS (SELECT parent_id, child_id AS d FROM e1),
             h2 AS (SELECT e1.parent_id, e2.child_id AS d
                    FROM e1 JOIN edges e2 ON e1.child_id=e2.parent_id),
             alld AS (SELECT parent_id, d FROM h1 UNION ALL SELECT parent_id, d FROM h2)
        SELECT parent_id, count(DISTINCT d) AS reach2
        FROM alld WHERE d <> parent_id GROUP BY parent_id;
    """)
    dt = time.time()-t0
    rs = con.execute("SELECT count(*), max(reach2), avg(reach2) FROM reach2_sample").fetchone()
    print(f"[reach2] sample of {rs[0]:,} parents computed in {dt:.1f}s  max reach2={rs[1]:,}  mean={rs[2]:.1f}")
    print("[reach2] top 10 in the sample:")
    for r in con.execute("""
        SELECT pk.name, pk.ecosystem, r.reach2
        FROM reach2_sample r JOIN packages pk ON TRY_CAST(pk.id AS BIGINT)=r.parent_id
        ORDER BY r.reach2 DESC LIMIT 10""").fetchall():
        print(f"    {r[0]!r:28} {r[1]!r:14} {r[2]:,}")

    n_parents = id_stats[0]
    print(f"\n[reach2] full graph has {n_parents:,} parents; sample was 2,000 "
          f"-> if linear, full exact reach2 ~ {dt * n_parents/2000/60:.1f} min "
          f"(hubs dominate super-linearly; the panel-repo restriction will cut this).")

    if args.validate:
        print("\nVALIDATE_REPORT_DONE")
        return

    con.execute("""
        CREATE OR REPLACE TABLE target_parents AS
        SELECT DISTINCT pkg_id FROM pkg_repo WHERE repo_norm IS NOT NULL AND pkg_id IS NOT NULL;
    """)
    ntp = con.execute("SELECT count(*) FROM target_parents").fetchone()[0]
    log(f"full build: reach2 for {ntp:,} github package-repo parents ...")
    t0 = time.time()
    con.execute("""
        CREATE OR REPLACE TABLE reach2 AS
        WITH e1 AS (SELECT e.parent_id, e.child_id
                    FROM edges e SEMI JOIN target_parents t ON t.pkg_id=e.parent_id),
             h1 AS (SELECT parent_id, child_id AS d FROM e1),
             h2 AS (SELECT e1.parent_id, e2.child_id AS d
                    FROM e1 JOIN edges e2 ON e1.child_id=e2.parent_id),
             alld AS (SELECT parent_id, d FROM h1 UNION ALL SELECT parent_id, d FROM h2)
        SELECT parent_id, count(DISTINCT d) AS reach2
        FROM alld WHERE d <> parent_id GROUP BY parent_id;
    """)
    log(f"reach2 done in {(time.time()-t0)/60:.1f} min")

    out_parq = os.path.join(OUT, f"ecosystems_{date}.parquet")
    con.execute(f"""
        COPY (
          SELECT pr.pkg_id AS package_id, pr.name AS package, pr.ecosystem,
                 pr.repo_norm, DATE '{date}' AS snapshot_date, 'ecosystems' AS source,
                 COALESCE(i.in_degree,0) AS in_degree,
                 COALESCE(r.reach2,0)   AS reach2
          FROM pkg_repo pr
          LEFT JOIN indeg  i ON i.parent_id = pr.pkg_id
          LEFT JOIN reach2 r ON r.parent_id = pr.pkg_id
          WHERE pr.repo_norm IS NOT NULL
        ) TO '{out_parq}' (FORMAT PARQUET, COMPRESSION ZSTD);
    """)
    stats = con.execute(f"""
        SELECT count(*) AS n_rows, count(DISTINCT repo_norm) AS repos,
               count(*) FILTER (WHERE reach2>0) AS with_reach, max(reach2) AS mx
        FROM read_parquet('{out_parq}')""").fetchone()
    log(f"wrote {out_parq}: {stats[0]:,} package-repo rows  distinct_repos={stats[1]:,}  "
        f"reach2>0={stats[2]:,}  max_reach2={stats[3]:,}")
    print("\nBUILD_DONE")

if __name__ == "__main__":
    main()
