import json, os
import numpy as np
import duckdb

import run_estimators as RE
import run_prepost as RP
import qmodel as QM

OUT = os.environ.get("CODERSQ_PHI_OUT", "/scratch/$USER/codersq/impact/estimates_v2/phi_dump.json")
SPLIT_SEED = 20260712

out = {"note": "anonymized plotting vectors, primary cells only; estimates unchanged",
       "split_seed": SPLIT_SEED, "grid_primary": {}, "prepost_primary": {}}

con = duckdb.connect()
con.execute("PRAGMA threads=4;")
rng = np.random.default_rng(SPLIT_SEED)
for half in ("discovery", "holdout"):
    p = RE.PRIMARY
    seqs, diag = RE.build_sequences(con, half, p["credit"], p["W"], p["n_min"], p["proxy"])
    pairs = []
    for seq in seqs.values():
        lc = np.log(np.asarray(seq, dtype=float))
        n = lc.size
        if n < 4:
            continue
        idx = rng.permutation(n)
        h = n // 2
        pairs.append([float(lc[idx[:h]].mean()), float(lc[idx[h:2 * h]].mean())])
    rng.shuffle(pairs)
    out["grid_primary"][half] = {"n_contributors": diag["n_contributors"],
                                 "split_half_pairs": pairs}
    print(f"grid {half}: {len(pairs)} split-half pairs (of {diag['n_contributors']} contributors)")
con.close()

def era_phi(con, era):
    rows = con.execute(f"""
        SELECT login, list(L ORDER BY first_ts, repo) AS Ls
        FROM realized_{era} GROUP BY login ORDER BY login;""").fetchall()
    logins = [lg for lg, _ in rows]
    seqlist = [np.exp(np.asarray(Ls, dtype=float)) for _, Ls in rows]
    if not seqlist:
        return {}
    fit = QM.fit_qmodel(seqlist)
    return {lg: float(q) for lg, q in zip(logins, np.asarray(fit["log_Q_hat"], dtype=float))}

for mode in ("real", "placebo"):
    out["prepost_primary"][mode] = {}
    for half in ("discovery", "holdout"):
        con = duckdb.connect()
        con.execute("PRAGMA threads=4;")
        parquet = RP.sql_literal(RP.WORK_REACH)
        RP.install_split_table(con, parquet)
        if mode == "placebo":
            RP.install_placebo_cohort(con, parquet)
        RP.prepare_cell_tables(con, parquet, mode, half, RP.PRIMARY)
        pre_phi, post_phi = era_phi(con, "pre"), era_phi(con, "post")
        common = sorted(set(pre_phi) & set(post_phi))
        pairs = [[pre_phi[lg], post_phi[lg]] for lg in common]
        rng.shuffle(pairs)
        out["prepost_primary"][mode][half] = {"n_pairs": len(pairs), "phi_pairs": pairs}
        print(f"prepost {mode}/{half}: {len(pairs)} phi pairs")
        con.close()

with open(OUT, "w") as f:
    json.dump(out, f)
print("wrote", OUT)
