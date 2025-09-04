#!/usr/bin/env python3
"""
Volatility plots for selected tickers (BlackRock-style), no intermediate files.

Reads:
  - analysis_scripts_2/data_files/top500_liquidity_2005_2023.csv  (permno, ticker)
  - analysis_scripts_2/data_files/vol_df.csv                      (permno, date, vol_hl*)

Outputs (PNG) ONLY to:
  - analysis_scripts_2/data_files/vol_plots/
"""

import os, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import AutoDateLocator, ConciseDateFormatter

# ===================== CONFIG =====================
LIQ_PATH = "analysis_scripts_2/data_files/top500_liquidity_2005_2023.csv"
VOL_PATH = "analysis_scripts_2/data_files/vol_df.csv"

# Focus tickers (case-insensitive)
FOCUS_TICKERS = ["BLK", "MSFT", "GOOGL"]

# Which half-life to feature in single-line plots & facets
HL_SHOW = 21

# Scale plots to annualized % (True) or daily % (False)
ANNUALIZE   = True
TRADING_DAYS = 252
SCALE        = (TRADING_DAYS**0.5 * 100.0) if ANNUALIZE else 100.0
SCALE_NAME   = "Annualized %" if ANNUALIZE else "Daily %"

# Where to save (ONLY here)
OUT_DIR = os.path.join("analysis_scripts_2", "data_files", "vol_plots")

# -------- Overlay period controls (choose ONE) --------
# Option A: explicit calendar window (recommended for slide)
OVERLAY_START = "2022-01-01"   # set to None to disable
OVERLAY_END   = "2023-12-31"   # set to None to disable

# Option B: fallback to last N calendar days if Option A is None
OVERLAY_LAST_N_DAYS = 365

# Figure sizes
FIGSIZE_TS        = (9, 3.6)        # single half-life TS
FIGSIZE_OVERLAY   = (12, 2.5)       # wide+short to fit bottom strip on slide
FIGSIZE_FACETS_H  = 1.9             # per row height (facets)
FIGSIZE_HIST      = (7.2, 4)
FIGSIZE_MEDIAN_HL = (7.2, 4)

# BlackRock-ish palette + global styling
PAL = {
    "gold":  "#FFD700",
    "orng":  "#FF4500",
    "black": "#000000",
    "gray":  "#6E6E6E",
    "muted": "#A9A9A9",
    "grid":  0.30,
}
plt.rcParams.update({
    "axes.edgecolor": PAL["black"],
    "axes.labelcolor": PAL["black"],
    "text.color":     PAL["black"],
    "xtick.color":    PAL["black"],
    "ytick.color":    PAL["black"],
    "font.size": 11,
    "axes.titlesize": 12,
    "figure.dpi": 110,
})

# ===================== HELPERS =====================
def ensure_dir(p): os.makedirs(p, exist_ok=True)

def figsave(path):
    ensure_dir(os.path.dirname(path))   # makes analysis_scripts_2/data_files/vol_plots if missing
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()

def date_axis(ax):
    loc = AutoDateLocator(minticks=6, maxticks=9)
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(ConciseDateFormatter(loc))

def volcols(df):  return [c for c in df.columns if re.match(r"vol_hl\d+$", c)]
def hl_from_col(c): return int(c.split("vol_hl")[-1])

def last_n_days(df, days=365, date_col="date", min_points=60):
    if df.empty:
        return df
    end = df[date_col].max()
    start = end - pd.Timedelta(days=days)
    w = df[(df[date_col] >= start) & (df[date_col] <= end)].copy()
    if len(w) < min_points:
        w = df.tail(252).copy()
    return w

def slice_between(df, start=None, end=None, date_col="date"):
    if start is None or end is None:
        return None
    s = pd.Timestamp(start)
    e = pd.Timestamp(end)
    return df[(df[date_col] >= s) & (df[date_col] <= e)].copy()

# ===================== LOAD & FILTER =====================
liq = pd.read_csv(LIQ_PATH, usecols=["permno", "ticker"])
liq["permno"] = liq["permno"].astype(int)
liq["ticker"] = liq["ticker"].astype(str).str.strip().str.upper()
liq = liq.drop_duplicates(subset=["permno", "ticker"])

df = pd.read_csv(VOL_PATH, parse_dates=["date"]).sort_values(["permno", "date"])
if "permno" not in df or "date" not in df:
    raise ValueError("vol_df.csv must have columns: permno, date, and vol_hl* columns.")
vcols = volcols(df)
if not vcols:
    raise ValueError("No vol_hl* columns found in vol_df.csv.")

df = df.merge(liq, on="permno", how="left")
focus = df[df["ticker"].isin([t.upper() for t in FOCUS_TICKERS])].copy()
if focus.empty:
    raise SystemExit("No rows matched FOCUS_TICKERS — check tickers or mapping file.")

# ===================== PANEL HISTOGRAM (HL_SHOW) =====================
col_show = f"vol_hl{HL_SHOW}"
if col_show not in focus.columns:
    raise ValueError(f"{col_show} not found. Available: {sorted(vcols)}")

plt.figure(figsize=FIGSIZE_HIST)
vals = (focus[col_show].values * SCALE)
vals = vals[np.isfinite(vals)]
plt.hist(vals, bins=80, color=PAL["muted"], edgecolor="white", linewidth=0.3)
plt.title(f"Distribution of EWMA vol (hl={HL_SHOW}) — focus tickers")
plt.xlabel(f"Vol ({SCALE_NAME})")
plt.ylabel("Frequency")
plt.grid(axis="y", alpha=PAL["grid"])
figsave(os.path.join(OUT_DIR, f"vol_hist_hl{HL_SHOW}_focus.png"))

# ===================== PER-TICKER PLOTS =====================
for tkr in FOCUS_TICKERS:
    dd = focus[focus["ticker"] == tkr].copy()
    if dd.empty:
        print(f"[{tkr}] no data — skipping.")
        continue

    pm_focus = int(dd["permno"].value_counts().idxmax())
    d0 = dd[dd["permno"] == pm_focus].copy()

    # --- 1) TS for single half-life (hl=HL_SHOW) ---
    plt.figure(figsize=FIGSIZE_TS)
    ax = plt.gca()
    ax.plot(d0["date"], d0[col_show]*SCALE, color=PAL["black"], lw=2)
    date_axis(ax)
    ax.set_title(f"{tkr} / PERMNO {pm_focus} — EWMA vol (hl={HL_SHOW})")
    ax.set_xlabel("Date")
    ax.set_ylabel(f"Vol ({SCALE_NAME})")
    ax.grid(alpha=PAL["grid"])
    figsave(os.path.join(OUT_DIR, f"{tkr}_permno{pm_focus}_vol_ts_hl{HL_SHOW}.png"))

    # --- 2) OVERLAY: multiple half-lives over custom window or last N days ---
    hls_available = sorted(hl_from_col(c) for c in vcols)
    overlay_pick = sorted(set([5, 10, 21, 63, 126]) & set(hls_available)) or hls_available[:5]
    colors = [PAL["gold"], PAL["orng"], PAL["black"], PAL["gray"], PAL["muted"]]

    dsel = slice_between(d0, OVERLAY_START, OVERLAY_END, "date")
    if dsel is None or dsel.empty:
        dsel = last_n_days(d0, days=OVERLAY_LAST_N_DAYS, date_col="date", min_points=60)

    start_lbl = dsel["date"].min().date()
    end_lbl   = dsel["date"].max().date()

    plt.figure(figsize=FIGSIZE_OVERLAY)   # wide + short to fit bottom white strip
    ax = plt.gca()
    handles, labels = [], []
    for i, hl in enumerate(overlay_pick):
        c = f"vol_hl{hl}"
        if c not in dsel.columns:
            continue
        h, = ax.plot(dsel["date"], dsel[c]*SCALE, lw=2, color=colors[i % len(colors)])
        handles.append(h); labels.append(f"hl={hl}")
    date_axis(ax)
    ax.set_title(f"{tkr} / PERMNO {pm_focus} — EWMA vol by half-life ({start_lbl} → {end_lbl})")
    ax.set_xlabel("Date")
    ax.set_ylabel(f"Vol ({SCALE_NAME})")
    ax.grid(alpha=PAL["grid"])
    if handles:
        ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.04),
                  ncol=len(labels), frameon=False, fontsize=9)
    plt.subplots_adjust(top=0.86)
    figsave(os.path.join(OUT_DIR, f"{tkr}_permno{pm_focus}_vol_overlay.png"))

    # --- 3) Facets across this ticker's PERMNOs (hl=HL_SHOW) ---
    permnos = dd["permno"].value_counts().index.tolist()
    N = min(6, len(permnos))
    fig, axes = plt.subplots(nrows=N, ncols=1, figsize=(9, FIGSIZE_FACETS_H*N), sharex=True)
    if N == 1: axes = [axes]
    for ax, pm in zip(axes, permnos[:N]):
        dpm = dd[dd["permno"] == pm]
        ax.plot(dpm["date"], dpm[col_show]*SCALE, color=PAL["black"], lw=1.7)
        ax.set_ylabel(f"{pm}\n{SCALE_NAME}", rotation=0, ha="right", va="center")
        ax.grid(alpha=PAL["grid"])
    date_axis(axes[-1])
    axes[0].set_title(f"{tkr} — EWMA vol (hl={HL_SHOW}), across PERMNOs")
    axes[-1].set_xlabel("Date")
    plt.tight_layout()
    figsave(os.path.join(OUT_DIR, f"{tkr}_vol_facets_hl{HL_SHOW}.png"))

    # --- 4) Median vol vs half-life (dominant PERMNO) ---
    meds = []
    for c in sorted(vcols, key=hl_from_col):
        meds.append((hl_from_col(c), np.nanmedian(d0[c]) * SCALE))
    meds = pd.DataFrame(meds, columns=["half_life", "median_vol"])

    plt.figure(figsize=FIGSIZE_MEDIAN_HL)
    plt.plot(meds["half_life"], meds["median_vol"], marker="o", lw=2, color=PAL["black"])
    plt.scatter(meds["half_life"], meds["median_vol"], s=40, color=PAL["gold"], edgecolors="white", zorder=3)
    plt.title(f"{tkr} / PERMNO {pm_focus} — Median EWMA vol vs half-life")
    plt.xlabel("Half-life (trading days)")
    plt.ylabel(f"Median vol ({SCALE_NAME})")
    plt.grid(alpha=PAL["grid"])
    figsave(os.path.join(OUT_DIR, f"{tkr}_permno{pm_focus}_vol_median_by_halflife.png"))

print(f"Done. Output dir: {OUT_DIR} | Scale={SCALE_NAME} | Overlay window:"
      f" {OVERLAY_START or 'last'} → {OVERLAY_END or f'last {OVERLAY_LAST_N_DAYS}d'}")
ç