#!/usr/bin/env python3
"""
Generates EPS feature plots (BlackRock-style palette):

  analysis_scripts_2/data_files/
    - eps_zscore_momentum_hist.png
    - eps_momentum_hist.png
    - {DEMO}_zscore_momentum_ts.png
    - {DEMO}_momentum_bars_last8.png
    - eps_dispersion_hist.png
    - {DEMO}_dispersion_ts.png

Set IN_PATH below if your CSV is elsewhere.
"""
#!/usr/bin/env python3
"""
EPS feature plots with aligned dates (BlackRock-style palette)

Outputs -> analysis_scripts_2/data_files/ :
  - eps_zscore_momentum_hist.png
  - eps_momentum_hist.png
  - {TICKER}_zscore_momentum_ts.png
  - {TICKER}_zscore_momentum_ts_aligned.png
  - eps_dispersion_hist.png
  - {TICKER}_dispersion_ts.png
  - {TICKER}_dispersion_ts_aligned.png
  - {TICKER}_momentum_bars_last8.png

Set IN_PATH / DEMO_TICKER / FORCE_CANONICAL_DATES below as needed.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import AutoDateLocator, ConciseDateFormatter

# =============== CONFIG ===============
IN_PATH   = "analysis_scripts_2/data_files/eps_features_at_analysis_dates.csv"  # path to your CSV
OUT_DIR   = "analysis_scripts_2/data_files/eps_plots"
DEMO_TICKER = "AAPL"      # e.g., "AAPL" to force; None -> auto choose most frequent

# (Optional) make analysis dates match canonical T*−BD_LAG with tolerance
FORCE_CANONICAL_DATES = True
BD_LAG   = 21          # business days before earnings
BD_TOL   = 1           # allow ±1 BD around the canonical date

# display-only clipping (to keep figures readable)
WINSOR = {
    "zscore": (-5, 5),
    "mom1m":  (-1, 1),
    "mom3m":  (-1.5, 1.5),
    "mom6m":  (-2, 2),
    "disp":   (0, 2),
}

# BlackRock-ish palette
def blkr_pal():
    return {
        "pre":   "#FFD700",  # gold (primary)
        "post":  "#FF4500",  # orange-red (secondary)
        "black": "#000000",  # main lines/text
        "lin":   "#6E6E6E",  # gray for baselines
        "muted": "#A9A9A9",  # light gray fills
        "grid":  0.30,
    }

pal = blkr_pal()

plt.rcParams.update({
    "axes.edgecolor": pal["black"],
    "axes.labelcolor": pal["black"],
    "text.color": pal["black"],
    "xtick.color": pal["black"],
    "ytick.color": pal["black"],
    "font.size": 11,
    "axes.titlesize": 12,
    "figure.dpi": 110,
})

# ===================== UTILS =====================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def figsave(path: str):
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()

def align_dates_for_ticker(df_tkr: pd.DataFrame, align_cols):
    """
    Drop rows with NaNs across align_cols and sort by analysis_date.
    Ensures the two series share identical x-axis dates.
    """
    d = df_tkr.copy()
    d = d.replace([np.inf, -np.inf], np.nan)
    d = d.dropna(subset=align_cols).sort_values("analysis_date")
    return d

def apply_canonical_filter(df: pd.DataFrame, lag_bd=21, tol_bd=1):
    """
    Keep rows whose analysis_date is within ±tol_bd business days of (earnings_date - lag_bd business days).
    """
    from pandas.tseries.offsets import BDay
    d = df.copy()
    d["canonical_analysis_date"] = d["earnings_date"] - BDay(lag_bd)
    # difference in days (calendar approximation is fine for tight tolerance)
    d["bdiff"] = (d["analysis_date"] - d["canonical_analysis_date"]).dt.days
    d = d[d["bdiff"].between(-tol_bd, tol_bd)]
    d = d.drop(columns=["canonical_analysis_date", "bdiff"])
    return d

# ===================== LOAD =====================
ensure_dir(OUT_DIR)
df = pd.read_csv(IN_PATH, parse_dates=["earnings_date", "analysis_date"], low_memory=False)
df = df.sort_values(["ticker", "analysis_date"])

need = {"ticker", "analysis_date", "earnings_date",
        "z_score_momentum", "momentum_1m", "momentum_3m", "momentum_6m",
        "dispersion_pct_ibes"}
missing = need - set(df.columns)
if missing:
    raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

if FORCE_CANONICAL_DATES:
    df_canon = apply_canonical_filter(df, lag_bd=BD_LAG, tol_bd=BD_TOL)
    if not df_canon.empty:
        df = df_canon  # enforce canonical schedule if available

# choose a demo ticker (most frequent by default)
if DEMO_TICKER is None:
    DEMO_TICKER = df["ticker"].value_counts().idxmax()

d = df[df["ticker"] == DEMO_TICKER].copy()
if d.empty:
    raise ValueError(f"No rows for DEMO_TICKER={DEMO_TICKER} after filtering.")

# ===================== PANEL PLOTS =====================

# 1) Panel distribution: z-score momentum
plt.figure(figsize=(7, 4))
z = df["z_score_momentum"].replace([np.inf, -np.inf], np.nan).dropna().clip(*WINSOR["zscore"])
plt.hist(z, bins=50, color=pal["muted"], edgecolor="white", linewidth=0.4)
plt.axvline(0, color=pal["lin"], ls="--", lw=1)
plt.title("EPS z-score momentum (panel-wide distribution)")
plt.xlabel("z-score of EPS consensus vs 1y history")
plt.ylabel("Frequency")
plt.grid(axis="y", alpha=pal["grid"])
figsave(os.path.join(OUT_DIR, "eps_zscore_momentum_hist.png"))

# 2) Panel distributions: 1m / 3m / 6m momentum
plt.figure(figsize=(7, 4))
m1 = df["momentum_1m"].replace([np.inf, -np.inf], np.nan).dropna().clip(*WINSOR["mom1m"])
m3 = df["momentum_3m"].replace([np.inf, -np.inf], np.nan).dropna().clip(*WINSOR["mom3m"])
m6 = df["momentum_6m"].replace([np.inf, -np.inf], np.nan).dropna().clip(*WINSOR["mom6m"])
plt.hist(m1, bins=60, alpha=0.55, color=pal["pre"],  label="1m", edgecolor="white", linewidth=0.3)
plt.hist(m3, bins=60, alpha=0.55, color=pal["post"], label="3m", edgecolor="white", linewidth=0.3)
plt.hist(m6, bins=60, alpha=0.45, color=pal["lin"],  label="6m", edgecolor="white", linewidth=0.3)
plt.title("EPS consensus momentum distributions")
plt.xlabel("Percent change")
plt.ylabel("Frequency")
plt.legend(frameon=False)
plt.grid(axis="y", alpha=pal["grid"])
figsave(os.path.join(OUT_DIR, "eps_momentum_hist.png"))

# ===================== TICKER PLOTS (unaligned) =====================

# 3) Ticker TS: z-score momentum (unaligned)
plt.figure(figsize=(8, 3.2))
ax = plt.gca()
ax.plot(d["analysis_date"], d["z_score_momentum"], color=pal["black"], lw=2)
ax.axhline(0, color=pal["lin"], ls="--", lw=1)
loc = AutoDateLocator(minticks=6, maxticks=9)
ax.xaxis.set_major_locator(loc); ax.xaxis.set_major_formatter(ConciseDateFormatter(loc))
plt.title(f"{DEMO_TICKER} — EPS z-score momentum over analysis dates")
plt.xlabel("Analysis date"); plt.ylabel("z-score momentum")
plt.grid(alpha=pal["grid"])
figsave(os.path.join(OUT_DIR, f"{DEMO_TICKER}_zscore_momentum_ts.png"))

# 4) Ticker TS: dispersion% (unaligned)
plt.figure(figsize=(8, 3.2))
ax = plt.gca()
ax.plot(d["analysis_date"], d["dispersion_pct_ibes"].replace([np.inf,-np.inf], np.nan), color=pal["black"], lw=2)
loc = AutoDateLocator(minticks=6, maxticks=9)
ax.xaxis.set_major_locator(loc); ax.xaxis.set_major_formatter(ConciseDateFormatter(loc))
plt.title(f"{DEMO_TICKER} — EPS estimate dispersion% over analysis dates")
plt.xlabel("Analysis date"); plt.ylabel("Dispersion%")
plt.grid(alpha=pal["grid"])
figsave(os.path.join(OUT_DIR, f"{DEMO_TICKER}_dispersion_ts.png"))

# 5) Ticker bars: last 8 analysis dates (1m / 3m / 6m)
tail = d.tail(8).copy()
x = np.arange(len(tail)); width = 0.28
plt.figure(figsize=(8, 3.6))
plt.bar(x - width, tail["momentum_1m"].values, width, color=pal["pre"],
        edgecolor="white", linewidth=0.4, label="1m")
plt.bar(x,         tail["momentum_3m"].values, width, color=pal["post"],
        edgecolor="white", linewidth=0.4, label="3m")
plt.bar(x + width, tail["momentum_6m"].values, width, color=pal["lin"],
        edgecolor="white", linewidth=0.4, label="6m")
plt.title(f"{DEMO_TICKER} — EPS momentum (last 8 analysis dates)")
plt.xlabel("Analysis date"); plt.ylabel("% change")
plt.xticks(x, [dt.strftime("%Y-%m-%d") for dt in tail["analysis_date"]], rotation=45, ha="right")
plt.legend(frameon=False)
plt.grid(axis="y", alpha=pal["grid"])
figsave(os.path.join(OUT_DIR, f"{DEMO_TICKER}_momentum_bars_last8.png"))

# ===================== TICKER PLOTS (ALIGNED X-AXES) =====================

ALIGN_COLS = ["z_score_momentum", "dispersion_pct_ibes"]
d_aligned = align_dates_for_ticker(d, ALIGN_COLS)

# 6) z-score momentum (aligned)
plt.figure(figsize=(8, 3.2))
ax = plt.gca()
ax.plot(d_aligned["analysis_date"], d_aligned["z_score_momentum"], color=pal["black"], lw=2)
ax.axhline(0, color=pal["lin"], ls="--", lw=1)
loc = AutoDateLocator(minticks=6, maxticks=9)
ax.xaxis.set_major_locator(loc); ax.xaxis.set_major_formatter(ConciseDateFormatter(loc))
plt.title(f"{DEMO_TICKER} — EPS z-score momentum (aligned dates)")
plt.xlabel("Analysis date"); plt.ylabel("z-score momentum")
plt.grid(alpha=pal["grid"])
figsave(os.path.join(OUT_DIR, f"{DEMO_TICKER}_zscore_momentum_ts_aligned.png"))

# 7) dispersion% (aligned — same x-axis as 6)
plt.figure(figsize=(8, 3.2))
ax = plt.gca()
ax.plot(d_aligned["analysis_date"], d_aligned["dispersion_pct_ibes"], color=pal["black"], lw=2)
loc = AutoDateLocator(minticks=6, maxticks=9)
ax.xaxis.set_major_locator(loc); ax.xaxis.set_major_formatter(ConciseDateFormatter(loc))
plt.title(f"{DEMO_TICKER} — EPS estimate dispersion% (aligned dates)")
plt.xlabel("Analysis date"); plt.ylabel("Dispersion%")
plt.grid(alpha=pal["grid"])
figsave(os.path.join(OUT_DIR, f"{DEMO_TICKER}_dispersion_ts_aligned.png"))

print(f"Saved plots in: {os.path.abspath(OUT_DIR)}")
print(f"Demo ticker: {DEMO_TICKER} | Canonical filter: {FORCE_CANONICAL_DATES} (lag={BD_LAG}, tol={BD_TOL})")

# === 8) Dual-axis plot: z-score momentum (LHS) vs dispersion (RHS) with date interval ===
from matplotlib.dates import AutoDateLocator, ConciseDateFormatter
from matplotlib.ticker import FuncFormatter

# Set your window here
START = pd.Timestamp("2012-01-01")
END   = pd.Timestamp("2022-12-31")

PAIR_ALIGN_COLS = ["z_score_momentum", "dispersion_pct_ibes"]
d_pair = align_dates_for_ticker(d, PAIR_ALIGN_COLS)

# Keep only dates in [START, END]
d_pair = d_pair[(d_pair["analysis_date"] >= START) & (d_pair["analysis_date"] <= END)]

if d_pair.empty:
    print(f"Skip dual-axis z-score vs dispersion: no overlapping dates in [{START.date()} … {END.date()}].")
else:
    fig, ax1 = plt.subplots(figsize=(9, 3.5))
    ax2 = ax1.twinx()

    # Left axis: z-score
    h1, = ax1.plot(
        d_pair["analysis_date"], d_pair["z_score_momentum"],
        color=pal["black"], lw=2, label="z-score momentum (LHS)"
    )
    ax1.axhline(0, color=pal["lin"], ls="--", lw=1)
    ax1.set_ylabel("z-score momentum")
    ax1.set_ylim(0, 4)

    # Right axis: dispersion
    h2, = ax2.plot(
        d_pair["analysis_date"], d_pair["dispersion_pct_ibes"],
        color=pal["pre"], lw=2, label="Dispersion (RHS)"
    )
    ax2.set_ylabel("Dispersion (%)")
    ax2.set_ylim(0, 0.15)   # <--- or (0, 15) if values are already in %
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{v*100:.0f}%"))
    if np.nanmax(d_pair["dispersion_pct_ibes"].values) <= 1.0:
        ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{v*100:.0f}%"))

    # Dates & limits
    loc = AutoDateLocator(minticks=6, maxticks=9)
    ax1.xaxis.set_major_locator(loc)
    ax1.xaxis.set_major_formatter(ConciseDateFormatter(loc))
    ax1.set_xlim(START, END)
    ax1.set_xlabel("Analysis date")
    ax1.set_title(f"{DEMO_TICKER} — EPS z-score momentum (LHS) & estimate dispersion (RHS)")

    # Grid & legend
    ax1.grid(alpha=pal["grid"])
    lines = [h1, h2]
    labels = [l.get_label() for l in lines]
    fig.legend(lines, labels, loc="upper center", bbox_to_anchor=(0.5, 1.06),
               ncol=2, frameon=False, fontsize=9)
    fig.subplots_adjust(top=0.88)

    out_path = os.path.join(OUT_DIR, f"{DEMO_TICKER}_z_vs_dispersion_dualaxis_{START.year}_{END.year}.png")
    figsave(out_path)
    print(f"Saved: {out_path}")
