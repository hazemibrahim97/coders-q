import json, os
import numpy as np
import duckdb
import run_estimators as RE

OUT = os.environ.get("CODERSQ_EXCAREERS_OUT",
                     "/scratch/$USER/codersq/impact/estimates_v2/example_careers.json")

con = duckdb.connect()
con.execute("PRAGMA threads=4;")
p = RE.PRIMARY
seqs, diag = RE.build_sequences(con, "discovery", p["credit"], p["W"], 10, p["proxy"])

cands = []
for login, seq in seqs.items():
    L = np.log(np.asarray(seq, dtype=float))
    n = L.size
    if not (12 <= n <= 20):
        continue
    spread = float(L.max() - L.min())
    peak_pos = (int(np.argmax(L)) + 1) / n
    cands.append((spread, peak_pos, L))
if not cands:
    raise SystemExit("no candidate careers")
spreads = sorted(c[0] for c in cands)
med = spreads[len(spreads) // 2]
cands = [c for c in cands if c[0] >= med]
picks = {}
for label, target in (("early", 0.1), ("mid", 0.5), ("late", 0.9)):
    best = min(cands, key=lambda c: abs(c[1] - target))
    picks[label] = {"peak_position": round(best[1], 3),
                    "n_works": int(best[2].size),
                    "L": [round(float(x), 4) for x in best[2]]}
    cands.remove(best)

with open(OUT, "w") as f:
    json.dump({"note": "anonymized exemplar careers, normalized log-impact by work order",
               "cell": {**p, "n_min": 10}, "careers": picks}, f)
print("wrote", OUT, {k: (v["n_works"], v["peak_position"]) for k, v in picks.items()})
