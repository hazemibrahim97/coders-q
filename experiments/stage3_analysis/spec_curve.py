import json, sys, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

path = sys.argv[1] if len(sys.argv) > 1 else "estimates_discovery_full.json"
outdir = sys.argv[sys.argv.index("--outdir") + 1] if "--outdir" in sys.argv else os.path.join(os.path.dirname(os.path.abspath(path)), "figs")
os.makedirs(outdir, exist_ok=True)

j = json.load(open(path))
rows = []
for r in j["results"]:
    if r.get("skipped"):
        continue
    c1, c2 = r["C1"], r["C2"]
    rows.append(dict(
        credit=r["credit"], W=r["W"], n_min=r["n_min"], proxy=r["proxy"],
        primary=r.get("primary", False), N=r["n_contributors"], frac_zero=r["frac_zero"],
        dep=c1["mean_P"] - c1["null_mean_P"], mean_P=c1["mean_P"], null_P=c1["null_mean_P"],
        tvd=c1["tvd"], perm_p=c1["perm_p"], bh=c1.get("bh_reject", None),
        A=c2["A_stableQ_share"], A_lo=c2["A_ci"][0], A_hi=c2["A_ci"][1],
        B=c2["B_AR_share"], ratio=c2["ratio_A_over_AB"], rho=c2["rho"],
        stationary=c2["stationary"], verdict=c2["verdict_stable_Q"],
        autocorr=c2.get("autocorr") or c2.get("r_k") or c2.get("autocorr_lag1to8"),
    ))
df = pd.DataFrame(rows)
live = df[df.W != 120]

SPEC_ROWS = [("credit", ["R-share", "R-lead", "R-core"]),
             ("W", sorted(df.W.unique())),
             ("n_min", sorted(df.n_min.unique())),
             ("proxy", sorted(df.proxy.unique()))]

def spec_matrix(ax, d):
    y = 0; yt, yl = [], []
    for field, levels in SPEC_ROWS:
        for lv in levels:
            on = (d[field] == lv).to_numpy()
            ax.scatter(np.where(on)[0], np.full(on.sum(), y), s=6,
                       c=["0.55" if w == 120 else "k" for w in d.W[on]])
            yt.append(y); yl.append(f"{field}={lv}")
            y -= 1
    ax.set_yticks(yt); ax.set_yticklabels(yl, fontsize=6)
    ax.set_xlim(-1, len(d)); ax.set_xlabel("specification (sorted)", fontsize=8)
    for s in ("top", "right"): ax.spines[s].set_visible(False)

d = df.sort_values("dep").reset_index(drop=True)
fig, (a0, a1) = plt.subplots(2, 1, figsize=(10, 7), height_ratios=[2.2, 1.6], sharex=True)
col = np.where(d.W == 120, "0.7", np.where(d.bh.fillna(False), "#c0392b", "#2c3e50"))
a0.bar(range(len(d)), d.dep, color=col, width=0.8)
a0.axhline(0, c="k", lw=0.8)
p = d.index[d.primary]
if len(p): a0.scatter(p, d.dep[p] + np.sign(d.dep[p]) * 0.002, marker="v", c="#f39c12", zorder=5, label="primary cell")
a0.set_ylabel("mean P − exact null (early-tilt < 0)")
a0.set_title("C1 random-impact: departure across the multiverse "
             "(red = BH-significant, grey = W120 degenerate)", fontsize=10)
a0.legend(fontsize=7, frameon=False)
spec_matrix(a1, d)
fig.tight_layout(); fig.savefig(f"{outdir}/spec_curve_C1.png", dpi=200); plt.close(fig)

d = df.sort_values("A").reset_index(drop=True)
fig, (a0, ar, a1) = plt.subplots(3, 1, figsize=(10, 8.5), height_ratios=[2.2, 1.0, 1.6], sharex=True)
col = np.where(d.W == 120, "0.7", np.where(d.verdict, "#27ae60", "#2c3e50"))
a0.errorbar(range(len(d)), d.A, yerr=[d.A - d.A_lo, d.A_hi - d.A], fmt="none", ecolor="0.6", lw=0.7)
a0.scatter(range(len(d)), d.A, c=col, s=14, zorder=3)
p = d.index[d.primary]
if len(p): a0.scatter(p, d.A[p], marker="v", c="#f39c12", s=60, zorder=4, label="primary cell")
a0.axhline(0, c="k", lw=0.8)
a0.set_ylabel("A (stable-Φ share)")
a0.set_title("C2 stable-Φ: A with CI (green = full gate passes, grey = W120)", fontsize=10)
a0.legend(fontsize=7, frameon=False)
ar.scatter(range(len(d)), d.ratio, c=col, s=10)
ar.axhline(0.20, c="#c0392b", lw=0.8, ls="--"); ar.set_ylabel("A/(A+B)", fontsize=8)
spec_matrix(a1, d)
fig.tight_layout(); fig.savefig(f"{outdir}/spec_curve_C2.png", dpi=200); plt.close(fig)

have_ac = df["autocorr"].map(lambda v: isinstance(v, (list, tuple)) and len(v) > 1)
if have_ac.any():
    fig, ax = plt.subplots(figsize=(7, 5))
    for _, r in df[have_ac].iterrows():
        ks = range(1, len(r["autocorr"]) + 1)
        c = "#f39c12" if r.primary else ("0.75" if r.W == 120 else "#2c3e50")
        ax.plot(ks, r["autocorr"], color=c,
                lw=2.2 if r.primary else 0.6, alpha=1.0 if r.primary else 0.5)
    ax.axhline(0, c="k", lw=0.8)
    ax.set_xlabel("lag k (works)"); ax.set_ylabel("r(k) within-person log-impact autocorr")
    ax.set_title("Raw autocorrelation: flat plateau = stable Φ, decay = momentum", fontsize=10)
    fig.tight_layout(); fig.savefig(f"{outdir}/autocorr_C2.png", dpi=200); plt.close(fig)

pr = df[df.primary]
def fmt(x): return f"{x:.3f}"
lines = [f"# Spec-curve summary — {j['half']} half, grid={j['grid']}",
         f"{len(df)} cells ({len(live)} live, {len(df)-len(live)} W120-degenerate). "
         f"N range {live.N.min():,}–{live.N.max():,}; zero-share {live.frac_zero.min():.2f}–{live.frac_zero.max():.2f}.",
         "", "## C1 random-impact",
         f"- departure (mean_P − null): median {fmt(live.dep.median())}, range [{fmt(live.dep.min())}, {fmt(live.dep.max())}] — all early-tilted" if (live.dep < 0).all() else
         f"- departure (mean_P − null): median {fmt(live.dep.median())}, range [{fmt(live.dep.min())}, {fmt(live.dep.max())}]",
         f"- BH-significant departure: {int(live.bh.fillna(False).sum())}/{len(live)} cells; TVD median {fmt(live.tvd.median())} (max {fmt(live.tvd.max())}) → law ≈holds w/ small tilt",
         "- by W: " + "; ".join(f"W{int(w)}: dep {fmt(g.dep.median())}, TVD {fmt(g.tvd.median())}" for w, g in live.groupby("W")),
         "", "## C2 stable-Φ",
         f"- A: median {fmt(live.A.median())}, range [{fmt(live.A.min())}, {fmt(live.A.max())}]; CI>0 in {int((live.A_lo > 0).sum())}/{len(live)} cells",
         f"- gate: verdict TRUE in {int(live.verdict.sum())}/{len(live)}; ratio≥0.20 in {int((live.ratio >= .2).sum())}; stationary in {int(live.stationary.sum())}",
         f"- by proxy: " + "; ".join(f"{p}: A {fmt(g.A.median())}, verdict {int(g.verdict.sum())}/{len(g)}" for p, g in live.groupby("proxy")),
         ""]
if len(pr):
    r = pr.iloc[0]
    lines += ["## Primary cell (R-share, W60, n_min 5, reach2)",
              f"- C1: mean_P {fmt(r.mean_P)} vs null {fmt(r.null_P)} (dep {fmt(r.dep)}), TVD {fmt(r.tvd)}, perm_p {r.perm_p:.2g}, BH {'REJECT' if r.bh else 'keep'}",
              f"- C2: A {fmt(r.A)} [{fmt(r.A_lo)}, {fmt(r.A_hi)}], ratio {fmt(r.ratio)}, stationary {r.stationary}, verdict {r.verdict}"]
open(f"{outdir}/spec_summary.md", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
print(f"\nfigures → {outdir}/")
