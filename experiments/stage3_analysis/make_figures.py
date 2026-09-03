from __future__ import annotations

import argparse
import json
import logging
import math
import os
from pathlib import Path
import tempfile
from typing import Any

_mpl_cache = Path(tempfile.gettempdir()) / "make_figures_mpl_cache"
_mpl_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_cache))

import matplotlib

logging.getLogger("fontTools").setLevel(logging.ERROR)
logging.getLogger("fontTools.ttLib").setLevel(logging.ERROR)
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results_v2"
OUTDIR = RESULTS / "figs_paper"
PAGE_W_IN = 183 / 25.4

OKABE_ITO = {
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "black": "#000000",
    "grey": "#8A8A8A",
    "lightgrey": "#D6D6D6",
}

def set_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 7,
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "lines.linewidth": 1.1,
            "patch.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )

def fail(message: str) -> None:
    raise SystemExit(f"make_figures.py: {message}")

def load_json(path: Path, *, optional: bool = False) -> dict[str, Any]:
    if not path.exists():
        if optional:
            return {}
        fail(f"missing required input file: {path.relative_to(ROOT)}")
    with path.open() as f:
        return json.load(f)

def load_optional_json(path: Path, warning: str) -> dict[str, Any] | None:
    if not path.exists():
        print(f"WARNING: {warning}")
        return None
    return load_json(path)

def live_results(data: dict[str, Any]) -> list[dict[str, Any]]:
    return [r for r in data.get("results", []) if not r.get("skipped")]

def primary_row(rows: list[dict[str, Any]], label: str) -> dict[str, Any]:
    prim = [r for r in rows if r.get("primary")]
    if len(prim) != 1:
        fail(f"expected exactly one primary cell in {label}; found {len(prim)}")
    return prim[0]

def safe_float(value: Any, default: float = np.nan) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

def p_text(value: Any) -> str:
    x = safe_float(value)
    if math.isnan(x):
        return "p=n/a"
    if x < 0.001:
        return f"p={x:.1e}"
    return f"p={x:.3f}"

def fmt(value: Any, digits: int = 3) -> str:
    x = safe_float(value)
    if math.isnan(x):
        return "n/a"
    return f"{x:.{digits}f}"

def pct(value: Any, digits: int = 1) -> str:
    x = safe_float(value)
    if math.isnan(x):
        return "n/a"
    return f"{100 * x:.{digits}f}%"

def nfmt(value: Any) -> str:
    x = safe_float(value)
    if math.isnan(x):
        return "n/a"
    return f"{int(round(x)):,}"

def share_vector(values: Any) -> np.ndarray:
    arr = np.asarray(values if isinstance(values, list) else [], dtype=float)
    total = arr.sum()
    if arr.size == 0 or not np.isfinite(total) or total <= 0:
        return np.asarray([], dtype=float)
    return arr / total

def tuple_array(data: dict[str, Any], key: str, width: int) -> np.ndarray:
    rows = data.get(key) or []
    out: list[list[float]] = []
    for row in rows:
        if not isinstance(row, list | tuple) or len(row) < width:
            continue
        vals = [safe_float(row[i]) for i in range(width)]
        if all(np.isfinite(vals)):
            out.append(vals)
    return np.asarray(out, dtype=float)

def pair_array(pairs: Any) -> np.ndarray:
    out: list[list[float]] = []
    for pair in pairs if isinstance(pairs, list) else []:
        if not isinstance(pair, list | tuple) or len(pair) < 2:
            continue
        x = safe_float(pair[0])
        y = safe_float(pair[1])
        if np.isfinite(x) and np.isfinite(y):
            out.append([x, y])
    return np.asarray(out, dtype=float)

def table_font(table: Any, size: float = 5.8, scale_y: float = 1.25) -> None:
    table.auto_set_font_size(False)
    table.set_fontsize(size)
    table.scale(1.0, scale_y)
    for (r, _c), cell in table.get_celld().items():
        cell.set_linewidth(0.45)
        cell.set_edgecolor("0.75")
        if r == 0:
            cell.set_facecolor("0.92")
            cell.set_text_props(weight="bold")

def identity_limits(arrays: list[np.ndarray]) -> tuple[float, float]:
    finite: list[np.ndarray] = []
    for arr in arrays:
        if arr.size:
            finite.append(arr[np.isfinite(arr)])
    if not finite:
        return 0.0, 1.0
    vals = np.concatenate(finite)
    if vals.size == 0:
        return 0.0, 1.0
    lo = float(np.nanmin(vals))
    hi = float(np.nanmax(vals))
    pad = max((hi - lo) * 0.06, 0.05)
    return lo - pad, hi + pad

def add_panel_label(ax: plt.Axes, letter: str) -> None:
    ax.text(
        -0.12,
        1.08,
        letter,
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        va="top",
        ha="left",
    )

def finish_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

def save_figure(fig: plt.Figure, stem: str) -> tuple[Path, Path]:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    pdf = OUTDIR / f"{stem}.pdf"
    png = OUTDIR / f"{stem}.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=300)
    plt.close(fig)
    return pdf, png

def estimates_dataframe(rows: list[dict[str, Any]]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for r in rows:
        c1 = r.get("C1") or {}
        c2 = r.get("C2") or {}
        ci = c2.get("A_ci") if isinstance(c2.get("A_ci"), list) else [np.nan, np.nan]
        records.append(
            {
                "credit": r.get("credit"),
                "W": r.get("W"),
                "n_min": r.get("n_min"),
                "proxy": r.get("proxy"),
                "primary": bool(r.get("primary")),
                "n_contributors": r.get("n_contributors"),
                "frac_zero": r.get("frac_zero"),
                "split_half_r": r.get("split_half_r"),
                "mean_P": c1.get("mean_P"),
                "null_mean_P": c1.get("null_mean_P"),
                "dep": safe_float(c1.get("mean_P")) - safe_float(c1.get("null_mean_P")),
                "tvd": c1.get("tvd"),
                "perm_p": c1.get("perm_p"),
                "bh_reject": bool(c1.get("bh_reject")),
                "A": c2.get("A_stableQ_share"),
                "A_lo": ci[0] if len(ci) > 0 else np.nan,
                "A_hi": ci[1] if len(ci) > 1 else np.nan,
                "B": c2.get("B_AR_share"),
                "ratio": c2.get("ratio_A_over_AB"),
                "rho": c2.get("rho"),
                "stationary": c2.get("stationary"),
                "verdict": bool(c2.get("verdict_stable_Q")),
                "autocorr": c2.get("autocorr_lag1to8"),
                "sigma_q2": (r.get("qfit") or {}).get("sigma_q2"),
                "sigma_p2": (r.get("qfit") or {}).get("sigma_p2"),
                "nbar_harmonic": (r.get("qfit") or {}).get("nbar_harmonic"),
            }
        )
    return pd.DataFrame.from_records(records)

def spec_levels(df: pd.DataFrame) -> list[tuple[str, list[Any]]]:
    order_credit = ["R-share", "R-lead", "R-core"]
    rows: list[tuple[str, list[Any]]] = []
    for field in ["credit", "W", "n_min", "proxy"]:
        vals = list(df[field].dropna().unique())
        if field == "credit":
            levels = [x for x in order_credit if x in vals] + sorted(
                [x for x in vals if x not in order_credit]
            )
        elif field in {"W", "n_min"}:
            levels = sorted(vals)
        else:
            levels = sorted(vals)
        rows.append((field, levels))
    return rows

def draw_spec_matrix(ax: plt.Axes, df: pd.DataFrame) -> None:
    y = 0
    yticks: list[int] = []
    labels: list[str] = []
    levels = spec_levels(df)
    for field, vals in levels:
        for value in vals:
            on = (df[field] == value).to_numpy()
            xs = np.flatnonzero(on)
            ax.scatter(xs, np.full(xs.size, y), s=7, color=OKABE_ITO["black"], linewidths=0)
            yticks.append(y)
            labels.append(f"{field}={value}")
            y -= 1
        if (field, vals) != levels[-1]:
            ax.axhline(y + 0.5, color="0.88", lw=0.5)
    ax.set_xlim(-1, len(df))
    ax.set_ylim(y + 0.4, 0.6)
    ax.set_yticks(yticks)
    ax.set_yticklabels(labels, fontsize=5.8)
    ax.tick_params(axis="x", length=0, labelsize=6)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("specification, sorted by mean P - null mean P")
    finish_axes(ax)

def plot_decile_panel(
    ax: plt.Axes,
    c1: dict[str, Any],
    *,
    title: str,
    show_legend: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    obs = share_vector(c1.get("counts"))
    exp = share_vector(c1.get("expected_counts"))
    if obs.size != 10 or exp.size != 10:
        fail(f"decile panel '{title}' needs 10 observed and expected bins")
    x = np.arange(1, 11)
    ax.bar(x, obs, width=0.72, color=OKABE_ITO["blue"], alpha=0.88, label="observed")
    ax.step(x, exp, where="mid", color=OKABE_ITO["orange"], lw=1.4, label="exact null")
    ax.set_title(title, fontsize=8, pad=3)
    ax.set_xlabel("impact decile")
    ax.set_ylabel("fraction of careers")
    ax.set_xticks(x)
    ax.set_xlim(0.35, 10.65)
    ax.set_ylim(0, max(float(obs.max()), float(exp.max())) * 1.22)
    if show_legend:
        ax.legend(frameon=False, fontsize=6, handlelength=1.6, loc="upper left")
    finish_axes(ax)
    return obs, exp

def make_fig1(desc: dict[str, Any]) -> str:
    career_panel = tuple_array(desc, "panel_career_len_hist", 2)
    career_matched = tuple_array(desc, "matched_career_len_hist", 2)
    reach = tuple_array(desc, "reach_hist_w60", 2)
    cohort_panel = tuple_array(desc, "panel_works_by_cohort", 2)
    cohort_matched = tuple_array(desc, "matched_works_by_cohort", 2)
    zero_by_w = tuple_array(desc, "zero_share_by_W", 3)
    censor_by_w = tuple_array(desc, "censor_flags_by_W", 3)
    if any(arr.size == 0 for arr in [career_panel, career_matched, reach, cohort_panel, cohort_matched, zero_by_w, censor_by_w]):
        fail("fig1_descriptives.json is missing one or more required descriptive arrays")

    fig = plt.figure(figsize=(PAGE_W_IN, 6.25), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1.0], width_ratios=[1.0, 1.0])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    sub_d = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1, 1], height_ratios=[1.1, 0.9], hspace=0.10)
    ax_d_line = fig.add_subplot(sub_d[0])
    ax_d_table = fig.add_subplot(sub_d[1])

    career_panel = career_panel[np.argsort(career_panel[:, 0])]
    career_matched = career_matched[np.argsort(career_matched[:, 0])]
    ax_a.bar(
        career_panel[:, 0],
        career_panel[:, 1],
        width=0.82,
        color=OKABE_ITO["lightgrey"],
        edgecolor="none",
        label="full panel",
    )
    ax_a.step(
        career_matched[:, 0],
        career_matched[:, 1],
        where="mid",
        color=OKABE_ITO["blue"],
        lw=1.4,
        label="matched W60",
    )
    ax_a.set_yscale("log")
    ax_a.set_xlabel("works per contributor")
    ax_a.set_ylabel("contributors")
    ax_a.set_xticks([1, 5, 10, 20, 30, 40, 50])
    ax_a.set_xticklabels(["1", "5", "10", "20", "30", "40", "50+"])
    ax_a.legend(frameon=False, fontsize=6, loc="upper right")
    finish_axes(ax_a)
    add_panel_label(ax_a, "a")

    reach = reach[np.argsort(reach[:, 0])]
    zero_mask = np.isclose(reach[:, 0], -1.0)
    zero_n = float(reach[zero_mask, 1].sum()) if zero_mask.any() else 0.0
    positive = reach[~zero_mask]
    reach_labels = ["zero"] + [f"{v:g}" for v in positive[:, 0]]
    reach_counts = np.concatenate([[zero_n], positive[:, 1]]) if zero_mask.any() else reach[:, 1]
    if not zero_mask.any():
        reach_labels = [f"{v:g}" for v in reach[:, 0]]
    x = np.arange(len(reach_counts))
    bars = ax_b.bar(x, reach_counts, color=OKABE_ITO["sky"], edgecolor="none", width=0.76)
    if zero_mask.any():
        bars[0].set_color(OKABE_ITO["grey"])
        bars[0].set_hatch("///")
    total_reach = float(np.nansum(reach[:, 1]))
    zero_share = zero_n / total_reach if total_reach else np.nan
    ax_b.set_yscale("log")
    ax_b.set_xlabel("log10 reach bin")
    ax_b.set_ylabel("works")
    tick_idx = np.linspace(0, len(reach_labels) - 1, min(8, len(reach_labels)), dtype=int)
    ax_b.set_xticks(tick_idx)
    ax_b.set_xticklabels([reach_labels[i] for i in tick_idx], rotation=0)
    ax_b.text(
        0.98,
        0.96,
        f"total works={nfmt(total_reach)}\nzero reach={pct(zero_share)}",
        transform=ax_b.transAxes,
        ha="right",
        va="top",
        fontsize=6.4,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.84, "pad": 2},
    )
    finish_axes(ax_b)
    add_panel_label(ax_b, "b")

    eco_rows = desc.get("eco_composition_w60") or []
    named: list[tuple[str, float, float]] = []
    other_works = 0.0
    other_logins = 0.0
    for row in eco_rows:
        if not isinstance(row, list | tuple) or len(row) < 3:
            continue
        eco = row[0]
        works = safe_float(row[1], 0.0)
        logins = safe_float(row[2], 0.0)
        if eco is None:
            other_works += works
            other_logins += logins
        else:
            named.append((str(eco), works, logins))
    named = sorted(named, key=lambda r: r[1], reverse=True)
    shown = named[:8]
    for _eco, works, logins in named[8:]:
        other_works += works
        other_logins += logins
    eco_plot = shown + [("other/unbridged", other_works, other_logins)]
    eco_plot = sorted(eco_plot, key=lambda r: r[1])
    y = np.arange(len(eco_plot))
    ax_c.barh(y, [r[1] for r in eco_plot], color=OKABE_ITO["green"], alpha=0.9)
    ax_c.set_yticks(y)
    ax_c.set_yticklabels([r[0] for r in eco_plot])
    ax_c.set_xlabel("works")
    for i, (_eco, works, _logins) in enumerate(eco_plot):
        ax_c.text(works * 1.01, i, nfmt(works), va="center", fontsize=5.8)
    ax_c.set_xlim(0, max(r[1] for r in eco_plot) * 1.22)
    finish_axes(ax_c)
    add_panel_label(ax_c, "c")

    cohort_panel = cohort_panel[np.argsort(cohort_panel[:, 0])]
    cohort_matched = cohort_matched[np.argsort(cohort_matched[:, 0])]
    ax_d_line.plot(cohort_panel[:, 0], cohort_panel[:, 1], marker="o", ms=2.7, color=OKABE_ITO["grey"], label="full panel")
    ax_d_line.plot(cohort_matched[:, 0], cohort_matched[:, 1], marker="s", ms=2.7, color=OKABE_ITO["blue"], label="matched W60")
    ax_d_line.set_yscale("log")
    ax_d_line.set_xlabel("cohort year")
    ax_d_line.set_ylabel("works")
    ax_d_line.legend(frameon=False, fontsize=6, loc="upper left")
    finish_axes(ax_d_line)
    add_panel_label(ax_d_line, "d")

    censor = {int(round(r[0])): (r[1], r[2]) for r in censor_by_w}
    table_rows: list[list[str]] = []
    for row in zero_by_w[np.argsort(zero_by_w[:, 0])]:
        w = int(round(row[0]))
        extrap, hole = censor.get(w, (np.nan, np.nan))
        note = "degenerate - dropped" if w == 120 else ""
        table_rows.append([str(w), pct(row[1], 1), pct(extrap, 1), pct(hole, 1), note])
    ax_d_table.axis("off")
    table = ax_d_table.table(
        cellText=table_rows,
        colLabels=["W", "zero", "right", "hole", "note"],
        loc="center",
        cellLoc="center",
        colLoc="center",
        colWidths=[0.10, 0.16, 0.16, 0.16, 0.33],
    )
    table_font(table, size=5.5, scale_y=1.18)

    save_figure(fig, "fig1")
    total_panel = float(np.nansum(career_panel[:, 1]))
    total_matched = float(np.nansum(career_matched[:, 1]))
    top_eco = max(shown, key=lambda r: r[1]) if shown else ("n/a", np.nan, np.nan)
    return (
        "Fig 1: "
        f"panel_contributors_hist_total={nfmt(total_panel)}, matched_W60_hist_total={nfmt(total_matched)}, "
        f"reach_total_works={nfmt(total_reach)}, zero_reach_share={fmt(zero_share)}, "
        f"top_ecosystem={top_eco[0]}:{nfmt(top_eco[1])}, W120_right_censored={pct(censor.get(120, [np.nan])[0])}"
    )

def make_fig2(est: dict[str, Any]) -> str:
    rows = live_results(est)
    if not rows:
        fail("estimates file has no live cells after skipped rows are removed")
    pr = primary_row(rows, "estimates")
    c1 = pr.get("C1") or {}
    obs = share_vector(c1.get("counts"))
    exp = share_vector(c1.get("expected_counts"))
    if obs.size != 10 or exp.size != 10:
        fail("primary estimates C1 has missing or malformed decile counts")

    df = estimates_dataframe(rows).sort_values("dep", kind="mergesort").reset_index(drop=True)
    primary_idx = int(df.index[df["primary"]].to_numpy()[0])

    fig = plt.figure(figsize=(PAGE_W_IN, 6.15), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[2.2, 1.38])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    sub = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1, :], height_ratios=[0.8, 1.55], hspace=0.06)
    ax_c_top = fig.add_subplot(sub[0])
    ax_c_mat = fig.add_subplot(sub[1], sharex=ax_c_top)

    plot_decile_panel(ax_a, c1, title="Primary cell: observed P_i vs exact null")
    ax_a.text(
        0.98,
        0.96,
        f"mean P={fmt(c1.get('mean_P'))}\nnull={fmt(c1.get('null_mean_P'))}\n"
        f"TVD={fmt(c1.get('tvd'))}\n{p_text(c1.get('perm_p')).replace('p=', 'perm p=')}",
        transform=ax_a.transAxes,
        ha="right",
        va="top",
        fontsize=6.6,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 2},
    )
    add_panel_label(ax_a, "a")

    x = np.arange(1, 11)
    cum_obs = np.cumsum(obs)
    cum_exp = np.cumsum(exp)
    gaps = np.abs(cum_obs - cum_exp)
    kmax = int(np.argmax(gaps))
    ax_b.plot(x, cum_obs, marker="o", ms=3, color=OKABE_ITO["blue"], label="observed")
    ax_b.plot(x, cum_exp, marker="s", ms=2.8, color=OKABE_ITO["orange"], label="exact null")
    ax_b.vlines(x[kmax], cum_obs[kmax], cum_exp[kmax], color=OKABE_ITO["vermillion"], lw=1.5)
    ax_b.annotate(
        f"max gap={gaps[kmax]:.3f}",
        xy=(x[kmax], (cum_obs[kmax] + cum_exp[kmax]) / 2),
        xytext=(0.50, 0.18),
        textcoords="axes fraction",
        arrowprops={"arrowstyle": "-", "lw": 0.7, "color": OKABE_ITO["vermillion"]},
        fontsize=6.6,
        color=OKABE_ITO["vermillion"],
    )
    ax_b.set_xlabel("impact decile")
    ax_b.set_ylabel("cumulative fraction")
    ax_b.set_xticks(x)
    ax_b.set_xlim(0.35, 10.65)
    ax_b.set_ylim(0, 1.02)
    ax_b.legend(frameon=False, fontsize=6, loc="upper left")
    finish_axes(ax_b)
    add_panel_label(ax_b, "b")

    colors = np.where(df["bh_reject"], OKABE_ITO["vermillion"], OKABE_ITO["grey"])
    ax_c_top.axhline(0, color="0.25", lw=0.7)
    ax_c_top.scatter(np.arange(len(df)), df["dep"], s=13, color=colors, linewidths=0, zorder=2)
    ax_c_top.scatter(
        [primary_idx],
        [df.loc[primary_idx, "dep"]],
        s=70,
        facecolors="none",
        edgecolors=OKABE_ITO["orange"],
        linewidths=1.4,
        zorder=4,
    )
    ax_c_top.annotate(
        "primary",
        xy=(primary_idx, df.loc[primary_idx, "dep"]),
        xytext=(5, 9),
        textcoords="offset points",
        fontsize=6.2,
        color=OKABE_ITO["orange"],
    )
    ax_c_top.set_ylabel("mean P - null")
    ax_c_top.tick_params(axis="x", labelbottom=False)
    ax_c_top.set_xlim(-1, len(df))
    ax_c_top.legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor=OKABE_ITO["vermillion"], markersize=4, label="BH reject"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=OKABE_ITO["grey"], markersize=4, label="BH keep"),
        ],
        frameon=False,
        fontsize=6,
        loc="upper left",
        ncol=2,
    )
    finish_axes(ax_c_top)
    add_panel_label(ax_c_top, "c")
    draw_spec_matrix(ax_c_mat, df)

    save_figure(fig, "fig2")
    return (
        "Fig 2: "
        f"primary mean_P={fmt(c1.get('mean_P'))}, null_mean_P={fmt(c1.get('null_mean_P'))}, "
        f"TVD={fmt(c1.get('tvd'))}, perm_p={fmt(c1.get('perm_p'), 4)}, "
        f"KS_gap={gaps[kmax]:.3f}, live_cells={len(rows)}, BH_reject={int(df['bh_reject'].sum())}"
    )

def draw_variance_budget(ax: plt.Axes, pr: dict[str, Any], c2: dict[str, Any], qfit: dict[str, Any]) -> tuple[float, float, float]:
    A = safe_float(c2.get("A_stableQ_share"))
    B = safe_float(c2.get("B_AR_share"))
    stable = max(0.0, min(1.0, A))
    ca_draw = max(0.0, min(B, 1.0 - stable))
    residual = max(0.0, 1.0 - stable - ca_draw)
    components = [
        ("stable A", stable, OKABE_ITO["green"]),
        ("CA (capped)", ca_draw, OKABE_ITO["orange"]),
        ("work-level luck", residual, OKABE_ITO["sky"]),
    ]
    left = 0.0
    for label, width, color in components:
        ax.barh([0], [width], left=[left], color=color, edgecolor="white", height=0.38, label=label)
        if width > 0.08:
            ax.text(left + width / 2, 0, f"{width:.2f}", ha="center", va="center", color="white", fontsize=6.6)
        left += width
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.58, 0.58)
    ax.set_yticks([])
    ax.set_xlabel("shares of within-person impact autocorrelation, not raw variance")
    ax.legend(frameon=False, fontsize=5.8, ncol=1, loc="upper left", bbox_to_anchor=(0.01, 0.99))
    ax.text(
        0.02,
        0.07,
        f"Raw B={fmt(B)}; sigma_q2={fmt(qfit.get('sigma_q2'))}, sigma_p2={fmt(qfit.get('sigma_p2'))}, "
        f"nbar_harmonic={fmt(qfit.get('nbar_harmonic'))}. "
        f"split-half r={fmt(pr.get('split_half_r'))}; zero-impact share={pct(pr.get('frac_zero'))}.",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.0,
        wrap=True,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.5},
    )
    finish_axes(ax)
    return stable, ca_draw, residual

def make_fig3(est: dict[str, Any], phi_dump: dict[str, Any] | None, half: str) -> str:
    rows = live_results(est)
    pr = primary_row(rows, "estimates")
    c2 = pr.get("C2") or {}
    qfit = pr.get("qfit") or {}
    autocorr = np.asarray(c2.get("autocorr_lag1to8") or [], dtype=float)
    if autocorr.size != 8:
        fail("primary estimates C2 has missing or malformed autocorr_lag1to8")

    df = estimates_dataframe(rows).sort_values("A", kind="mergesort").reset_index(drop=True)
    primary_idx = int(df.index[df["primary"]].to_numpy()[0])

    fig = plt.figure(figsize=(PAGE_W_IN, 6.35), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0], width_ratios=[1.0, 1.0])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    k = np.arange(1, 9)
    A = safe_float(c2.get("A_stableQ_share"))
    B = safe_float(c2.get("B_AR_share"))
    rho = safe_float(c2.get("rho"))
    k_fit = np.linspace(1, 8, 180)
    fit = A + B * np.power(rho, k_fit)
    ax_a.axhspan(0, A, color=OKABE_ITO["green"], alpha=0.12, zorder=0)
    ax_a.axhline(A, color=OKABE_ITO["green"], lw=1.0, zorder=1)
    ax_a.plot(k_fit, fit, color=OKABE_ITO["orange"], lw=1.5, label="A + B rho^k", zorder=3)
    ax_a.plot(
        k,
        autocorr,
        "o",
        color=OKABE_ITO["blue"],
        markeredgecolor=OKABE_ITO["blue"],
        ms=3.8,
        alpha=1.0,
        label="observed r(k)",
        zorder=4,
    )
    ax_a.text(
        0.03,
        0.06,
        f"A={fmt(A)} [{fmt((c2.get('A_ci') or [np.nan, np.nan])[0])}, {fmt((c2.get('A_ci') or [np.nan, np.nan])[1])}]\n"
        f"A/(A+B)={fmt(c2.get('ratio_A_over_AB'))}\nstationary={c2.get('stationary')}",
        transform=ax_a.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.6,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.84, "pad": 2},
    )
    ax_a.annotate(
        "stable share A",
        xy=(8.0, A),
        xytext=(4.7, A + 0.055),
        arrowprops={"arrowstyle": "-", "lw": 0.7, "color": OKABE_ITO["green"]},
        fontsize=6.4,
        color=OKABE_ITO["green"],
    )
    ax_a.set_xlabel("lag k (works)")
    ax_a.set_ylabel("within-person impact autocorrelation")
    ax_a.set_xticks(k)
    ax_a.legend(frameon=False, fontsize=6, loc="lower right")
    finish_axes(ax_a)
    add_panel_label(ax_a, "a")

    y = np.arange(len(df))
    verdict_colors = np.where(df["verdict"], OKABE_ITO["green"], OKABE_ITO["grey"])
    xerr = np.vstack(
        [
            np.maximum(0, df["A"].astype(float).to_numpy() - df["A_lo"].astype(float).to_numpy()),
            np.maximum(0, df["A_hi"].astype(float).to_numpy() - df["A"].astype(float).to_numpy()),
        ]
    )
    ax_b.errorbar(df["A"], y, xerr=xerr, fmt="none", ecolor="0.72", lw=0.7, zorder=1)
    ax_b.scatter(df["A"], y, s=14, color=verdict_colors, linewidths=0, zorder=2)
    ax_b.scatter(
        [df.loc[primary_idx, "A"]],
        [primary_idx],
        s=76,
        facecolors="none",
        edgecolors=OKABE_ITO["orange"],
        linewidths=1.4,
        zorder=4,
    )
    ax_b.axvline(0, color="0.2", lw=0.7)
    ax_b.set_xlabel("stable share A")
    ax_b.set_ylabel("specification")
    ax_b.set_yticks([])
    ax_b.set_ylim(-1, len(df))
    ax_b2 = ax_b.twiny()
    ax_b2.scatter(df["ratio"], y, marker="|", s=42, color=OKABE_ITO["purple"], linewidths=1.0, zorder=3)
    ax_b2.axvline(0.20, color=OKABE_ITO["purple"], lw=0.8, ls=":")
    ax_b2.set_xlabel("A/(A+B)")
    ax_b2.tick_params(axis="x", labelsize=6)
    ax_b.legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor=OKABE_ITO["green"], markersize=4, label="stable-Q verdict"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=OKABE_ITO["grey"], markersize=4, label="not full verdict"),
            Line2D([0], [0], marker="|", color=OKABE_ITO["purple"], markersize=7, label="ratio marker"),
        ],
        frameon=False,
        fontsize=6,
        loc="lower right",
    )
    finish_axes(ax_b)
    ax_b2.spines["top"].set_visible(True)
    ax_b2.spines["right"].set_visible(False)
    add_panel_label(ax_b, "b")

    split_obj = ((phi_dump or {}).get("grid_primary") or {}).get(half) or {}
    split_pairs = pair_array(split_obj.get("split_half_pairs"))
    if split_pairs.size:
        lo, hi = identity_limits([split_pairs])
        ax_c.scatter(
            split_pairs[:, 0],
            split_pairs[:, 1],
            s=2,
            alpha=0.15,
            color=OKABE_ITO["blue"],
            linewidths=0,
            rasterized=True,
        )
        ax_c.plot([lo, hi], [lo, hi], color="0.25", lw=0.8)
        ax_c.set_xlim(lo, hi)
        ax_c.set_ylim(lo, hi)
        ax_c.text(
            0.03,
            0.97,
            f"r={fmt(pr.get('split_half_r'))}\nn={nfmt(split_obj.get('n_contributors') or len(split_pairs))}",
            transform=ax_c.transAxes,
            ha="left",
            va="top",
            fontsize=6.4,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.84, "pad": 2},
        )
        ax_c.text(
            0.03,
            0.03,
            f"independent random within-career split (seed {(phi_dump or {}).get('split_seed')}); "
            "r from the sealed run",
            transform=ax_c.transAxes,
            ha="left",
            va="bottom",
            fontsize=5.8,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.5},
        )
        scatter_note = f"split_pairs={len(split_pairs)}"
    else:
        ax_c.text(
            0.5,
            0.5,
            "split-half pairs unavailable",
            transform=ax_c.transAxes,
            ha="center",
            va="center",
            fontsize=7,
            color="0.35",
        )
        scatter_note = "split_pairs=unavailable"
    ax_c.set_xlabel("mean log-impact, half A")
    ax_c.set_ylabel("mean log-impact, half B")
    finish_axes(ax_c)
    add_panel_label(ax_c, "c")

    stable, ca_draw, residual = draw_variance_budget(ax_d, pr, c2, qfit)
    add_panel_label(ax_d, "d")

    save_figure(fig, "fig3")
    return (
        "Fig 3: "
        f"primary A={fmt(A)}, A_CI=[{fmt((c2.get('A_ci') or [np.nan, np.nan])[0])}, "
        f"{fmt((c2.get('A_ci') or [np.nan, np.nan])[1])}], B_raw={fmt(B)}, "
        f"rho={fmt(rho)}, ratio={fmt(c2.get('ratio_A_over_AB'))}, "
        f"stationary={c2.get('stationary')}, verdict={c2.get('verdict_stable_Q')}, "
        f"split_half_r={fmt(pr.get('split_half_r'))}, {scatter_note}, "
        f"variance_budget=[A={stable:.3f}, CA_capped={ca_draw:.3f}, residual={residual:.3f}], "
        f"frac_zero={fmt(pr.get('frac_zero'))}"
    )

def load_prepost() -> dict[str, dict[str, Any]]:
    files = {
        "real-discovery": RESULTS / "prepost_real_discovery.json",
        "real-holdout": RESULTS / "prepost_real_holdout.json",
        "placebo-discovery": RESULTS / "prepost_placebo_discovery.json",
        "placebo-holdout": RESULTS / "prepost_placebo_holdout.json",
    }
    return {name: load_json(path) for name, path in files.items()}

def era_c2(row: dict[str, Any], era: str) -> dict[str, Any]:
    return ((row.get("per_era") or {}).get(era) or {}).get("C2") or {}

def era_c1(row: dict[str, Any], era: str) -> dict[str, Any]:
    return ((row.get("per_era") or {}).get(era) or {}).get("C1") or {}

def ci_pair(c2: dict[str, Any]) -> tuple[float, float]:
    ci = c2.get("A_ci") if isinstance(c2.get("A_ci"), list) else [np.nan, np.nan]
    lo = safe_float(ci[0] if len(ci) > 0 else np.nan)
    hi = safe_float(ci[1] if len(ci) > 1 else np.nan)
    return lo, hi

def draw_phi_summary_strip(ax: plt.Axes, primary: dict[str, dict[str, Any]], order: list[str]) -> None:
    colors = [
        to_rgba(OKABE_ITO["blue"] if name.startswith("real") else OKABE_ITO["grey"], 1.0 if name.startswith("real") else 0.62)
        for name in order
    ]
    rhos = [safe_float((primary[name].get("phi_rank_corr") or {}).get("rho")) for name in order]
    ax.axhline(0, color="0.25", lw=0.7)
    ax.bar(np.arange(len(order)), rhos, color=colors, width=0.58)
    ax.scatter(np.arange(len(order)), rhos, color="white", edgecolor="0.2", s=16, zorder=3, linewidths=0.7)
    for i, rho in enumerate(rhos):
        ax.text(i, rho + 0.025, fmt(rho), ha="center", va="bottom", fontsize=5.5)
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels([s.replace("-", "\n") for s in order], fontsize=5.4)
    ax.set_ylabel("rho", fontsize=6)
    ymax = max([x for x in rhos if np.isfinite(x)] + [0]) + 0.12
    ax.set_ylim(0, ymax)
    finish_axes(ax)

def draw_phi_scatter(
    ax: plt.Axes,
    primary: dict[str, dict[str, Any]],
    phi_dump: dict[str, Any] | None,
    prepost_half: str,
) -> str:
    real_pairs = pair_array((((phi_dump or {}).get("prepost_primary") or {}).get("real") or {}).get(prepost_half, {}).get("phi_pairs"))
    placebo_pairs = pair_array((((phi_dump or {}).get("prepost_primary") or {}).get("placebo") or {}).get(prepost_half, {}).get("phi_pairs"))
    real_phi = primary[f"real-{prepost_half}"].get("phi_rank_corr") or {}
    placebo_phi = primary[f"placebo-{prepost_half}"].get("phi_rank_corr") or {}

    if real_pairs.size:
        lo, hi = identity_limits([real_pairs, placebo_pairs])
        ax.scatter(
            real_pairs[:, 0],
            real_pairs[:, 1],
            s=8,
            alpha=0.36,
            color=OKABE_ITO["blue"],
            linewidths=0,
            rasterized=True,
        )
        ax.plot([lo, hi], [lo, hi], color="0.25", lw=0.8)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.text(
            0.03,
            0.97,
            f"real {prepost_half}\nrho={fmt(real_phi.get('rho'))}, n={nfmt(real_phi.get('n'))}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=6.2,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.84, "pad": 2},
        )
        if placebo_pairs.size:
            inset = ax.inset_axes([0.61, 0.08, 0.36, 0.36])
            inset.scatter(
                placebo_pairs[:, 0],
                placebo_pairs[:, 1],
                s=7,
                alpha=0.34,
                color=OKABE_ITO["grey"],
                linewidths=0,
                rasterized=True,
            )
            inset.plot([lo, hi], [lo, hi], color="0.35", lw=0.6)
            inset.set_xlim(lo, hi)
            inset.set_ylim(lo, hi)
            inset.set_xticks([])
            inset.set_yticks([])
            inset.set_title(f"placebo rho={fmt(placebo_phi.get('rho'))}, n={nfmt(placebo_phi.get('n'))}", fontsize=5.2, pad=1.5)
            finish_axes(inset)
        note = f"phi_pairs_real_{prepost_half}={len(real_pairs)}, phi_pairs_placebo_{prepost_half}={len(placebo_pairs)}"
    else:
        ax.text(
            0.5,
            0.5,
            "phi-pair vectors unavailable",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=7,
            color="0.35",
        )
        note = f"phi_pairs_{prepost_half}=unavailable"
    ax.set_xlabel("Phi-hat pre-era")
    ax.set_ylabel("Phi-hat post-era")
    finish_axes(ax)
    return note

def make_fig4(prepost: dict[str, dict[str, Any]], phi_dump: dict[str, Any] | None, prepost_half: str) -> str:
    primary: dict[str, dict[str, Any]] = {}
    for name, data in prepost.items():
        primary[name] = primary_row(live_results(data), name)

    real_holdout = primary.get("real-holdout") or {}
    pre_c1 = era_c1(real_holdout, "pre")
    post_c1 = era_c1(real_holdout, "post")

    fig = plt.figure(figsize=(PAGE_W_IN, 6.65), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1.0], width_ratios=[1.05, 0.95])
    sub_a = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0, 0], wspace=0.16)
    ax_a1 = fig.add_subplot(sub_a[0])
    ax_a2 = fig.add_subplot(sub_a[1], sharey=ax_a1)
    ax_b = fig.add_subplot(gs[0, 1])
    sub_c = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1, 0], height_ratios=[1.2, 0.48], hspace=0.24)
    ax_c = fig.add_subplot(sub_c[0])
    ax_c_strip = fig.add_subplot(sub_c[1])
    ax_d = fig.add_subplot(gs[1, 1])

    obs_pre, exp_pre = plot_decile_panel(ax_a1, pre_c1, title="pre era", show_legend=True)
    obs_post, exp_post = plot_decile_panel(ax_a2, post_c1, title="post era", show_legend=False)
    ytop = max(ax_a1.get_ylim()[1], ax_a2.get_ylim()[1])
    ax_a1.set_ylim(0, ytop)
    ax_a2.set_ylim(0, ytop)
    ax_a2.set_ylabel("")
    ax_a2.tick_params(axis="y", labelleft=False)
    for ax, c1, era in [(ax_a1, pre_c1, "pre"), (ax_a2, post_c1, "post")]:
        era_data = (real_holdout.get("per_era") or {}).get(era) or {}
        ax.text(
            0.98,
            0.96,
            f"mean P={fmt(c1.get('mean_P'))}\nTVD={fmt(c1.get('tvd'))}\n"
            f"{p_text(c1.get('perm_p')).replace('p=', 'perm p=')}\nworks={nfmt(era_data.get('n_works'))}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=6,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.84, "pad": 1.8},
        )
    add_panel_label(ax_a1, "a")

    order = ["real-discovery", "real-holdout", "placebo-discovery", "placebo-holdout"]
    y = np.arange(len(order))[::-1]
    for i, name in enumerate(order):
        row = primary[name]
        pre = era_c2(row, "pre")
        post = era_c2(row, "post")
        pre_a = safe_float(pre.get("A_stableQ_share"))
        post_a = safe_float(post.get("A_stableQ_share"))
        pre_lo, pre_hi = ci_pair(pre)
        post_lo, post_hi = ci_pair(post)
        is_real = name.startswith("real")
        color = OKABE_ITO["blue"] if is_real else OKABE_ITO["grey"]
        lw = 1.8 if is_real else 1.0
        alpha = 1.0 if is_real else 0.62
        yy = y[i]
        ax_b.plot([pre_a, post_a], [yy, yy], color=color, lw=lw, alpha=alpha, zorder=1)
        ax_b.errorbar(
            [pre_a],
            [yy],
            xerr=[[max(0, pre_a - pre_lo)], [max(0, pre_hi - pre_a)]],
            fmt="o",
            color=color,
            ecolor=color,
            ms=4 if is_real else 3.2,
            lw=0.9,
            alpha=alpha,
            zorder=3,
        )
        ax_b.errorbar(
            [post_a],
            [yy],
            xerr=[[max(0, post_a - post_lo)], [max(0, post_hi - post_a)]],
            fmt="s",
            color=color,
            ecolor=color,
            ms=4 if is_real else 3.2,
            lw=0.9,
            alpha=alpha,
            zorder=3,
        )
        ax_b.text(
            0.985,
            yy,
            f"Delta={post_a - pre_a:+.3f}",
            transform=ax_b.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=6.2,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.0},
        )
    ax_b.set_yticks(y)
    ax_b.set_yticklabels([s.replace("-", "\n") for s in order])
    ax_b.set_xlabel("stable share A")
    ax_b.set_title("A before and after", fontsize=8, pad=3)
    ax_b.legend(
        handles=[
            Line2D([0], [0], marker="o", color="0.25", label="pre", markersize=4, lw=0),
            Line2D([0], [0], marker="s", color="0.25", label="post", markersize=4, lw=0),
            Patch(facecolor=OKABE_ITO["blue"], label="real"),
            Patch(facecolor=OKABE_ITO["grey"], label="placebo"),
        ],
        frameon=False,
        fontsize=6,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=2,
    )
    finish_axes(ax_b)
    add_panel_label(ax_b, "b")

    phi_note = draw_phi_scatter(ax_c, primary, phi_dump, prepost_half)
    add_panel_label(ax_c, "c")
    draw_phi_summary_strip(ax_c_strip, primary, order)

    ax_d.axis("off")
    rows_table = []
    for name in ["real-discovery", "real-holdout"]:
        row = primary[name]
        eras = row.get("per_era") or {}
        pre = eras.get("pre") or {}
        post = eras.get("post") or {}
        rows_table.append(
            [
                name.replace("real-", ""),
                nfmt(row.get("n_incumbents")),
                nfmt(pre.get("n_works")),
                nfmt(post.get("n_works")),
                pct(row.get("frac_extrap_right_post"), 1),
            ]
        )
    table = ax_d.table(
        cellText=rows_table,
        colLabels=["half", "incumbents", "pre works", "post works", "post right-censored"],
        loc="center",
        cellLoc="center",
        colLoc="center",
        colWidths=[0.18, 0.22, 0.2, 0.2, 0.26],
    )
    table_font(table, size=5.9, scale_y=1.35)
    ax_d.set_title("primary real-cell censoring disclosure", fontsize=8, pad=4)
    add_panel_label(ax_d, "d")

    save_figure(fig, "fig4")
    deltas = []
    for name in order:
        row = primary[name]
        pre_a = safe_float(era_c2(row, "pre").get("A_stableQ_share"))
        post_a = safe_float(era_c2(row, "post").get("A_stableQ_share"))
        deltas.append(f"{name}_DeltaA={post_a - pre_a:+.3f}")
    return (
        "Fig 4: "
        f"real-holdout pre mean_P={fmt(pre_c1.get('mean_P'))}, post mean_P={fmt(post_c1.get('mean_P'))}, "
        f"pre_TVD={fmt(pre_c1.get('tvd'))}, post_TVD={fmt(post_c1.get('tvd'))}, "
        f"scatter_half={prepost_half}, frozen_real_phi_rho={fmt((primary[f'real-{prepost_half}'].get('phi_rank_corr') or {}).get('rho'))}, "
        f"frozen_real_phi_n={nfmt((primary[f'real-{prepost_half}'].get('phi_rank_corr') or {}).get('n'))}, "
        f"{phi_note}; "
        + ", ".join(deltas)
    )

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--half",
        choices=["discovery", "holdout"],
        default="discovery",
        help="Which estimates_{half}_full.json file feeds Fig. 2 and Fig. 3.",
    )
    parser.add_argument(
        "--prepost-half",
        choices=["discovery", "holdout"],
        default="holdout",
        help="Which pre/post half feeds the Fig. 4 phi-pair scatter.",
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    set_style()

    estimates_path = RESULTS / f"estimates_{args.half}_full.json"
    estimates = load_json(estimates_path)
    prepost = load_prepost()
    descriptives = load_optional_json(
        RESULTS / "fig1_descriptives.json",
        "missing optional input results_v2/fig1_descriptives.json; skipping Fig 1.",
    )
    phi_dump = load_optional_json(
        RESULTS / "phi_dump.json",
        "missing optional input results_v2/phi_dump.json; split-half and phi-pair scatter panels will use placeholders.",
    )

    summaries = []
    if descriptives is not None:
        summaries.append(make_fig1(descriptives))
    summaries.extend(
        [
            make_fig2(estimates),
            make_fig3(estimates, phi_dump, args.half),
            make_fig4(prepost, phi_dump, args.prepost_half),
        ]
    )
    for line in summaries:
        print(line)

if __name__ == "__main__":
    main()
