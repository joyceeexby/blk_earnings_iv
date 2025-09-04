"""
Generates:
  - {ticker}_{analysis_date}_iv_tte_ievr.png
  - {ticker}_{analysis_date}_iv_tte_kink.png
  - {ticker}_{analysis_date}_atm_oi_snapshot.png
  - {ticker}_{analysis_date}_atm_oi_mean_pre5.png   (if available)
  - {ticker}_{analysis_date}_atm_spread_ts.png
Saved under analysis_scripts_2/data_files/
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wrds

from matplotlib.ticker import MaxNLocator, AutoMinorLocator
from matplotlib.dates import AutoDateLocator, ConciseDateFormatter

# --- your modules ---
from ievr_batch_runner import (
    IVSurfaceCache,
    fetch_iv_surface_for_date,
    get_secid_at_date,
)
from ievr_analysis import IEVRAnalysis

# ========================
# CONFIG
# ========================
ticker = "GOOG"
earnings_date = pd.Timestamp("2022-04-26")
analysis_days_before = 21
use_business_days = True

FIGSIZE_MAIN = (7, 6)

# Filters consistent with runner / IEVR
ATM_LOW, ATM_HIGH = 0.95, 1.05
POST_PLUS_DAYS    = 30
TTE_MIN, TTE_MAX  = 7, 120

OUT_DIR = "analysis_scripts_2/data_files"

# ========================
# Utilities
# ========================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def figsave(path):
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()

def palette():
    return {
        "raw":    "#000000",  # black
        "pre":    "#FFD700",  # yellow
        "post":   "#FF4500",  # orange-red
        "global": "#000000",  # black for global fit (dash-dot)
        "event":  "#000000",  # black
        "lin":    "#6E6E6E",  # grey for linear baselines
        "unused": "#A9A9A9",  # dim gray for unused ATM points
        "grid":   0.3,
    }

def pick_opprcd_table(db, year):
    q = """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'optionm' AND table_name LIKE 'opprcd%%'
    """
    avail = set(db.raw_sql(q)["table_name"].str.lower())
    if f"opprcd{year}" in avail:
        return f"optionm.opprcd{year}"
    if "opprcd" in avail:
        return "optionm.opprcd"
    raise RuntimeError("No optionm.opprcd* table found.")

def fetch_quotes_atm(db, secid, start_date, end_date):
    """ATM quotes panel for OI / spread charts."""
    tbl = pick_opprcd_table(db, pd.to_datetime(start_date).year)
    q = f"""
        SELECT date, exdate, strike_price, cp_flag,
               best_bid, best_offer, volume, open_interest, impl_volatility
        FROM {tbl}
        WHERE secid = {secid}
          AND date BETWEEN '{start_date}' AND '{end_date}'
          AND impl_volatility > 0 AND impl_volatility < 5.0
          AND best_bid >= 0 AND best_offer >= 0
        ORDER BY date, exdate, strike_price
    """
    df = db.raw_sql(q)
    if df.empty:
        return df

    df["date"]   = pd.to_datetime(df["date"])
    df["exdate"] = pd.to_datetime(df["exdate"])

    # Underlying for moneyness
    px = db.raw_sql(f"""
        SELECT date, close
        FROM optionm.secprd
        WHERE secid = {secid} AND date BETWEEN '{start_date}' AND '{end_date}'
    """)
    px["date"] = pd.to_datetime(px["date"])
    df = df.merge(px, on="date", how="left", validate="many_to_one")

    df.rename(columns={"close": "underlying"}, inplace=True)
    df["underlying"] = df["underlying"].ffill().bfill()
    df["moneyness"]  = (df["strike_price"] / 1000.0) / df["underlying"]
    df["tte"]        = (df["exdate"] - df["date"]).dt.days

    # ATM + sensible TTE
    df = df[
        df["moneyness"].between(ATM_LOW, ATM_HIGH) &
        df["tte"].between(TTE_MIN, TTE_MAX)
    ].copy()

    # Spread %
    mid = (df["best_bid"] + df["best_offer"]) / 2.0
    spr = (df["best_offer"] - df["best_bid"])
    df["spread_pct"] = np.where(mid > 0, spr / mid, np.nan)

    return df

# ========================
# Dates
# ========================
if use_business_days:
    from pandas.tseries.offsets import BDay
    analysis_date = earnings_date - BDay(analysis_days_before)
else:
    analysis_date = earnings_date - pd.Timedelta(days=analysis_days_before)

ensure_dir(OUT_DIR)
pal = palette()

# ========================
# WRDS pulls
# ========================
db = wrds.Connection()

# A) IV surface for the IV-vs-TTE & kink plots (closest quote day)
cache = IVSurfaceCache()
iv_surface, closest_date = fetch_iv_surface_for_date(
    db=db,
    ticker=ticker,
    analysis_date=analysis_date,
    cache=cache,
    window_days=30
)

if iv_surface is None or iv_surface.empty:
    db.close()
    raise RuntimeError("No IV surface returned; try different dates/ticker.")

title_date = closest_date.date() if pd.notna(closest_date) else "N/A"

# ATM slice + PRE/POST(+30) windows
atm = iv_surface[iv_surface["moneyness"].between(ATM_LOW, ATM_HIGH)].copy()
days_to_earnings = int((earnings_date - analysis_date).days)
pre  = atm[atm["tte"] <  days_to_earnings].copy()
post = atm[(atm["tte"] > days_to_earnings) & (atm["tte"] <= days_to_earnings + POST_PLUS_DAYS)].copy()
used = pd.concat([pre, post], ignore_index=True).sort_values("tte")
unused = atm.loc[~atm.index.isin(pre.index) & ~atm.index.isin(post.index)].copy()

# B) Quotes panel for OI/spread charts — extend to earnings_date
secid = get_secid_at_date(db, ticker, analysis_date)
if secid is None:
    db.close()
    raise RuntimeError(f"No secid for {ticker} as of {analysis_date.date()}")

ts_start = (analysis_date - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
ts_end   = earnings_date.strftime("%Y-%m-%d")    # extended to T*
ts_df = fetch_quotes_atm(db, secid, ts_start, ts_end)

db.close()

# ========================
# Analyzer & helpers
# ========================
analyzer = IEVRAnalysis(db_connection=None)

def smooth_window(df, n=200, bandwidth_scale=0.6):
    df = df.dropna(subset=["tte", "put_iv"]).sort_values("tte")
    tte = df["tte"].values
    ivs = df["put_iv"].values
    if len(tte) < 2:
        return tte, ivs
    grid = np.linspace(tte.min(), tte.max(), n)
    bw = np.std(tte) * bandwidth_scale if len(tte) > 1 else 1.0
    sm = [analyzer.kernel_regression_iv(tte, ivs, t, bandwidth=bw) for t in grid]
    return grid, np.array(sm)

def kernel_avg_iv_runner_style(df):
    df = df.dropna(subset=["tte", "put_iv"])
    if df.empty:
        return np.nan
    tte = df["tte"].values; ivs = df["put_iv"].values
    if len(df) == 1:
        return float(ivs[0])
    grid = np.linspace(df["tte"].min(), df["tte"].max(), max(5, min(10, len(df))))
    vals = [analyzer.kernel_regression_iv(tte, ivs, t) for t in grid]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if vals else np.nan

def tidy_x_axis(ax, xmin, xmax):
    pad = max(2.0, 0.04 * (xmax - xmin))  # ~4% or ≥2d on each side
    ax.set_xlim(xmin - pad, xmax + pad)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    plt.margins(x=0.02)

# ========================
# 1) IV vs TTE (show ALL ATM incl. 80–120d; fits/IEVR use PRE+POST)
# ========================
if len(pre) >= 1 and len(post) >= 1:
    # Kernel fits (windows)
    pre_grid,  pre_smooth  = smooth_window(pre)
    post_grid, post_smooth = smooth_window(post)

    # Global kernel over ALL ATM
    full_grid = np.linspace(atm["tte"].min(), atm["tte"].max(), 250)
    global_smooth = np.array([
        analyzer.kernel_regression_iv(atm["tte"].values, atm["put_iv"].values, t)
        for t in full_grid
    ])

    # Runner-style averages + IEVR
    avg_pre  = kernel_avg_iv_runner_style(pre)
    avg_post = kernel_avg_iv_runner_style(post)
    ievr = (avg_post / avg_pre) if (np.isfinite(avg_pre) and avg_pre > 0 and np.isfinite(avg_post)) else np.nan

    # Piecewise linear on windows
    m_pre = b_pre = m_post = b_post = None
    if len(pre)  >= 2: m_pre,  b_pre  = np.polyfit(pre["tte"].values,  pre["put_iv"].values,  1)
    if len(post) >= 2: m_post, b_post = np.polyfit(post["tte"].values, post["put_iv"].values, 1)
    line_x = np.linspace(atm["tte"].min(), atm["tte"].max(), 250)
    pre_line_y  = (m_pre  * line_x + b_pre)  if m_pre  is not None else None
    post_line_y = (m_post * line_x + b_post) if m_post is not None else None

    # Plot
    plt.figure(figsize=FIGSIZE_MAIN)

    # All ATM points (unused in grey), with crisp edges
    if not unused.empty:
        plt.scatter(
            unused["tte"], unused["put_iv"]*100,
            s=18, alpha=0.22, color=pal["unused"],
            edgecolors="white", linewidths=0.25, zorder=2,
            label="ATM (unused)"
        )
    plt.scatter(
        pre["tte"], pre["put_iv"]*100,
        s=32, alpha=0.9, color=pal["pre"],
        edgecolors="white", linewidths=0.25, zorder=3,
        label="Pre (used)"
    )
    plt.scatter(
        post["tte"], post["put_iv"]*100,
        s=32, alpha=0.9, color=pal["post"],
        edgecolors="white", linewidths=0.25, zorder=3,
        label="Post (used)"
    )

    # Kernel fits
    plt.plot(pre_grid,  pre_smooth*100,  lw=2.0, color=pal["pre"],  label="Pre Kernel Fit")
    plt.plot(post_grid, post_smooth*100, lw=2.0, color=pal["post"], label="Post Kernel Fit")
    # plt.plot(full_grid, global_smooth*100, lw=2.0, ls="-.", color=pal["global"], label="Global Kernel Fit (ATM)")

    # Piecewise linear
    if pre_line_y is not None:
        plt.plot(line_x, pre_line_y*100,  ls=":",  lw=1.8, color=pal["lin"], label="Pre linear trend (OLS)")
    if post_line_y is not None:
        plt.plot(line_x, post_line_y*100, ls="--", lw=1.4, color=pal["lin"], label="Post linear trend (OLS)")

    # Event & averages
    plt.axvline(days_to_earnings, color=pal["event"], ls="--", lw=1.3, label="Earnings (T*)")
    if np.isfinite(avg_pre):
        plt.axhline(avg_pre*100,  ls="--", lw=1.3, color=pal["pre"],  label="Pre Kernel Avg")
    if np.isfinite(avg_post):
        plt.axhline(avg_post*100, ls="--", lw=1.3, color=pal["post"], label="Post Kernel Avg")

    # Crisp x-axis
    ax = plt.gca()
    tidy_x_axis(ax, float(atm["tte"].min()), float(atm["tte"].max()))

    plt.title(f"{ticker} — IV vs TTE (Closest quote: {title_date}); IEVR ≈ {ievr:.3f}", fontweight="bold")
    plt.xlabel("Time to Expiration (Days)"); plt.ylabel("Implied Volatility (%)")
    plt.legend(loc="lower right", framealpha=0.95)
    plt.grid(alpha=pal["grid"])
    out_path = os.path.join(OUT_DIR, f"{ticker}_{analysis_date.date()}_iv_tte_ievr.png")
    figsave(out_path)
    print(f"Saved: {out_path}")
else:
    print("Skip IV vs TTE plot: not enough pre/post ATM points.")

# ========================
# 2) Kink motivation plot (piecewise linear, PRE/POST used)
# ========================
try:
    m_pre = b_pre = m_post = b_post = None
    if len(pre)  >= 2: m_pre,  b_pre  = np.polyfit(pre["tte"].values,  pre["put_iv"].values,  1)
    if len(post) >= 2: m_post, b_post = np.polyfit(post["tte"].values, post["put_iv"].values, 1)
    line_x2 = np.linspace(used["tte"].min(), used["tte"].max(), 250)
    pre_lin_y2  = (m_pre  * line_x2 + b_pre)  if m_pre  is not None else None
    post_lin_y2 = (m_post * line_x2 + b_post) if m_post is not None else None

    plt.figure(figsize=FIGSIZE_MAIN)
    plt.scatter(
        pre["tte"],  pre["put_iv"]*100,  s=30, alpha=0.9, color=pal["pre"],
        edgecolors="white", linewidths=0.25, zorder=3, label="Pre (used)"
    )
    plt.scatter(
        post["tte"], post["put_iv"]*100, s=30, alpha=0.9, color=pal["post"],
        edgecolors="white", linewidths=0.25, zorder=3, label="Post (used)"
    )

    if pre_lin_y2 is not None:
        plt.plot(line_x2, pre_lin_y2*100,  ls=":",  lw=1.8, color=pal["lin"], label="Pre linear trend (OLS)")
    if post_lin_y2 is not None:
        plt.plot(line_x2, post_lin_y2*100, ls="--", lw=1.4, color=pal["lin"], label="Post linear trend (OLS)")

    plt.axvline(days_to_earnings, color=pal["event"], ls="--", lw=1.3, label="Earnings (T*)")

    # Crisp x-axis (use used-window range)
    ax = plt.gca()
    tidy_x_axis(ax, float(used["tte"].min()), float(used["tte"].max()))

    plt.title(f"{ticker} — Earnings 'kink' in ATM IV (Closest quote: {title_date})", fontweight="bold")
    plt.xlabel("Time to Expiration (Days)"); plt.ylabel("Implied Volatility (%)")
    plt.legend(loc="lower right", framealpha=0.95)
    plt.grid(alpha=pal["grid"])
    out_path = os.path.join(OUT_DIR, f"{ticker}_{analysis_date.date()}_iv_tte_kink.png")
    figsave(out_path)
    print(f"Saved: {out_path}")
except Exception as e:
    print(f"Skip kink plot due to error: {e}")

# ========================
# 3) ATM Open Interest charts
# ========================
if not ts_df.empty:
    # Closest date snapshot (closest to analysis_date)
    closest_idx = (ts_df["date"] - analysis_date).abs().idxmin()
    closest_qdate = ts_df.loc[closest_idx, "date"].normalize()

    snap = ts_df[ts_df["date"] == closest_qdate].copy()
    oi_by_type = (snap.groupby("cp_flag")["open_interest"]
                    .sum()
                    .rename(index={"P": "Puts", "C": "Calls"})
                    .reindex(["Puts", "Calls"], fill_value=0))

    plt.figure(figsize=FIGSIZE_MAIN)
    bars = plt.bar(oi_by_type.index, oi_by_type.values, width=0.55, color=[pal["pre"], pal["post"]])
    plt.title(f"{ticker} — ATM Open Interest by Type\nClosest quote: {closest_qdate.date()}")
    plt.ylabel("Open Interest (contracts)")
    for b in bars:
        plt.text(b.get_x()+b.get_width()/2, b.get_height()*1.01, f"{int(b.get_height()):,}",
                 ha="center", va="bottom", fontsize=10)
    plt.grid(axis="y", alpha=pal["grid"])
    out_path = os.path.join(OUT_DIR, f"{ticker}_{analysis_date.date()}_atm_oi_snapshot.png")
    figsave(out_path)
    print(f"Saved: {out_path}")

    # Mean of last 5 calendar days before analysis date
    pre_mask = (ts_df["date"] < analysis_date) & (ts_df["date"] >= analysis_date - pd.Timedelta(days=5))
    if pre_mask.any():
        oi_pre = (ts_df.loc[pre_mask].groupby(["date", "cp_flag"])["open_interest"]
                    .sum()
                    .reset_index()
                    .pivot(index="date", columns="cp_flag", values="open_interest")
                    .rename(columns={"P": "Puts", "C": "Calls"}))
        plt.figure(figsize=FIGSIZE_MAIN)
        oi_pre.mean().reindex(["Puts", "Calls"]).plot(kind="bar",
                                                      color=[pal["pre"], pal["post"]],
                                                      width=0.55)
        plt.title(f"{ticker} — Avg ATM Open Interest (Last 5 days pre analysis date)")
        plt.ylabel("Open Interest (contracts)")
        plt.grid(axis="y", alpha=pal["grid"])
        out_path = os.path.join(OUT_DIR, f"{ticker}_{analysis_date.date()}_atm_oi_mean_pre5.png")
        figsave(out_path)
        print(f"Saved: {out_path}")
    else:
        print("Skip OI mean chart: not enough days before analysis date.")
else:
    print("Skip OI charts: no ATM quotes in panel window.")

# ========================
# 4) ATM Bid–Ask Spread % time series (extended to earnings, banner legend)
# ========================
if not ts_df.empty:
    ts = (ts_df.groupby(["date", "cp_flag"])["spread_pct"]
            .median()
            .reset_index()
            .pivot(index="date", columns="cp_flag", values="spread_pct")
            .rename(columns={"P": "Puts", "C": "Calls"}))

    fig, ax = plt.subplots(figsize=(9,3.5)) #CHANGE HERE

    handles = []
    if "Puts" in ts:
        h1, = ax.plot(ts.index, ts["Puts"]*100, lw=2.0, color=pal["pre"],
                      label="Puts — median spread (% of mid)")
        handles.append(h1)
    if "Calls" in ts:
        h2, = ax.plot(ts.index, ts["Calls"]*100, lw=2.0, color=pal["post"],
                      label="Calls — median spread (% of mid)")
        handles.append(h2)

    h3 = ax.axvline(analysis_date, color="#444444", ls="--", lw=1.0, label="Analysis date (T−x)")
    h4 = ax.axvline(earnings_date, color=pal["event"], ls="-.", lw=1.2, label="Earnings date (T*)")
    handles.extend([h3, h4])

    # Compact, readable dates
    loc = AutoDateLocator(minticks=6, maxticks=9)
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(ConciseDateFormatter(loc))
    ax.tick_params(axis="x", rotation=0)

    ax.set_title(f"{ticker} — ATM Bid–Ask Spread by Type (median, % of mid)\nWindow: {ts_start} … {ts_end}")
    ax.set_ylabel("Median spread (% of mid)")
    ax.set_xlabel("Trade date")
    ax.grid(alpha=pal["grid"])

    # One-row banner legend above the axes
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.02),
               ncol=len(handles), frameon=False, fontsize=9, handlelength=2.6, columnspacing=1.6)
    fig.subplots_adjust(top=0.86)

    out_path = os.path.join(OUT_DIR, f"{ticker}_{analysis_date.date()}_atm_spread_ts.png")
    figsave(out_path)
    print(f"Saved: {out_path}")
else:
    print("Skip Spread TS: no ATM quotes in panel window.")

# ========================
# 5) Combined chart: Kink (top) + Spread TS (bottom)
# ========================
# if len(pre) >= 1 and len(post) >= 1 and not ts_df.empty:
#     fig, (ax1, ax2) = plt.subplots(
#         nrows=2, ncols=1, figsize=(9, 7),
#         gridspec_kw={'height_ratios': [2.5, 1]}
#     )

#     # --- ax1: Kink plot ---
#     ax1.scatter(pre["tte"], pre["put_iv"]*100, s=28, color=pal["pre"], label="Pre (used)")
#     ax1.scatter(post["tte"], post["put_iv"]*100, s=28, color=pal["post"], label="Post (used)")
#     ax1.axvline(days_to_earnings, ls="--", color="black", lw=1.3, label="Earnings (T*)")
#     ax1.set_title(f"{ticker} — Earnings 'kink' in ATM IV (Closest quote: {title_date})")
#     ax1.set_ylabel("Implied Volatility (%)")
#     ax1.grid(alpha=0.3)

#     # --- ax2: Spread TS plot ---
#     if "Puts" in ts:
#         ax2.plot(ts.index, ts["Puts"]*100, lw=2.0, color=pal["pre"], label="Puts — median spread %")
#     if "Calls" in ts:
#         ax2.plot(ts.index, ts["Calls"]*100, lw=2.0, color=pal["post"], label="Calls — median spread %")
#     ax2.axvline(analysis_date, color="#444444", ls="--", lw=1.0, label="Analysis date (T−x)")
#     ax2.axvline(earnings_date, color="black", ls="-.", lw=1.2, label="Earnings date (T*)")
#     ax2.set_ylabel("Spread (% of mid)")
#     ax2.set_xlabel("Trade Date")
#     ax2.grid(alpha=0.3)

#     # Legend at bottom
#     ax2.legend(loc="lower center", bbox_to_anchor=(0.5, -0.45),
#                ncol=3, frameon=False, fontsize=9)
#     ax1.set_xlabel("Time to Expiration (Days)")
#     ax2.set_xlabel("Trade Date")
#     fig.subplots_adjust(hspace=0.35, bottom=0.18)
#     # fig.subplots_adjust(hspace=0.3, bottom=0.22)

#     out_path = os.path.join(OUT_DIR, f"{ticker}_{analysis_date.date()}_kink_plus_spread.png")
#     figsave(out_path)
#     print(f"Saved: {out_path}")
# else:
#     print("Skip combined kink+spread chart: not enough data.")

# ========================
# 5) Combined chart: Kink (top) + Spread TS (bottom)
# ========================
if len(pre) >= 1 and len(post) >= 1 and not ts_df.empty:
    fig, (ax1, ax2) = plt.subplots(
        nrows=2, ncols=1, figsize=(9, 7),
        gridspec_kw={'height_ratios': [2.5, 1]}
    )

    # --- ax1: Kink plot ---
    ax1.scatter(pre["tte"], pre["put_iv"]*100, s=28, color=pal["pre"], label="Pre (used)")
    ax1.scatter(post["tte"], post["put_iv"]*100, s=28, color=pal["post"], label="Post (used)")
    ax1.axvline(days_to_earnings, ls="--", color="black", lw=1.3, label="Earnings (T*)")
    ax1.set_title(f"{ticker} — Earnings 'kink' in ATM IV (Closest quote: {title_date})")
    ax1.set_ylabel("Implied Volatility (%)")
    ax1.set_xlabel("Time to Expiration (Days)")
    ax1.grid(alpha=0.3)

    # --- ax2: Spread TS plot ---
    if "Puts" in ts:
        ax2.plot(ts.index, ts["Puts"]*100, lw=2.0, color=pal["pre"], label="Puts — median spread %")
    if "Calls" in ts:
        ax2.plot(ts.index, ts["Calls"]*100, lw=2.0, color=pal["post"], label="Calls — median spread %")
    ax2.axvline(analysis_date, color="#444444", ls="--", lw=1.0, label="Analysis date (T−x)")
    ax2.axvline(earnings_date, color="black", ls="-.", lw=1.2, label="Earnings date (T*)")
    ax2.set_ylabel("Spread (% of mid)")
    # no x-label → ticks alone are enough
    ax2.grid(alpha=0.3)

    # Legend at bottom, single row
    ax2.legend(loc="lower center", bbox_to_anchor=(0.5, -0.45),
               ncol=3, frameon=False, fontsize=9)

    # Adjust spacing (smaller bottom since no x-label)
    fig.subplots_adjust(hspace=0.35, bottom=0.15)

    out_path = os.path.join(OUT_DIR, f"{ticker}_{analysis_date.date()}_kink_plus_spread.png")
    figsave(out_path)
    print(f"Saved: {out_path}")
else:
    print("Skip combined kink+spread chart: not enough data.")


# ========================
# 6) ATM Open Interest near earnings (avg of last N business days before T*)
# ========================
N_BDAYS_BEFORE_E = 5  # change if you want a different window

if not ts_df.empty:
    from pandas.tseries.offsets import BDay

    bd_end   = (earnings_date - BDay(1)).normalize()
    bd_start = (earnings_date - BDay(N_BDAYS_BEFORE_E)).normalize()

    mask_e = (ts_df["date"] >= bd_start) & (ts_df["date"] <= bd_end)
    ts_near_e = ts_df.loc[mask_e].copy()

    if not ts_near_e.empty:
        # sum OI per day/type, then average across the business days window
        daily_oi = (ts_near_e.groupby(["date", "cp_flag"])["open_interest"]
                               .sum()
                               .reset_index())
        avg_oi = (daily_oi.pivot(index="date", columns="cp_flag", values="open_interest")
                           .rename(columns={"P": "Puts", "C": "Calls"}))
        avg_vals = avg_oi.mean().reindex(["Puts", "Calls"]).fillna(0)

        plt.figure(figsize=FIGSIZE_MAIN)
        bars = plt.bar(["Puts", "Calls"], avg_vals.values,
                       width=0.55, color=[pal["pre"], pal["post"]])
        plt.title(
            f"{ticker} — Avg ATM Open Interest (last {N_BDAYS_BEFORE_E} business days before earnings)\n"
            f"Window: {bd_start.date()} … {bd_end.date()}"
        )
        plt.ylabel("Open Interest (contracts)")
        for b, v in zip(bars, avg_vals.values):
            plt.text(b.get_x() + b.get_width()/2, v*1.01, f"{int(v):,}",
                     ha="center", va="bottom", fontsize=10)
        plt.grid(axis="y", alpha=pal["grid"])

        out_path = os.path.join(
            OUT_DIR,
            f"{ticker}_{analysis_date.date()}_atm_oi_mean_preE_{N_BDAYS_BEFORE_E}bd.png"
        )
        figsave(out_path)
        print(f"Saved: {out_path}")
    else:
        print(f"Skip OI near-earnings chart: no ATM quotes between {bd_start.date()} and {bd_end.date()}.")
else:
    print("Skip OI near-earnings chart: no ATM quotes in panel window.")
