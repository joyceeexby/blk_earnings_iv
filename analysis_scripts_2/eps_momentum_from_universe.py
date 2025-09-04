"""
EPS Features at Analysis Dates (21 BDays Pre-Earnings) — Ticker + CUSIP Only
https://wrds-www.wharton.upenn.edu/pages/get-data/lseg-ibes/ibes-academic/summary-history/summary-statistics/ - table
-----------------------------------------------------------------------------
- Loads earnings dates: analysis_scripts_2/data_files/earnings_dates.csv
  Required: earnings_date. Optional: cusip / cusip8 / ticker.
- Loads universe (fallback IDs): analysis_scripts_2/data_files/top500_liquidity_2005_2023.csv
- Pulls IBES from tr_ibes.statsum_epsus by (CUSIP OR IBES ticker)
- Computes: meanest, stdev, numest, dispersion_pct, momentum_1m/3m/6m,
            rolling_momentum_3m, z_score_momentum
- For each (ticker/cusip8, earnings_date): analysis_date = earnings_date - 21 BDays
  As-of match last statpers <= analysis_date
- Writes one long table:
    analysis_scripts_2/data_files/eps_features_at_analysis_dates.csv
  Columns:
    cusip8, ticker, earnings_date, analysis_date, matched_statpers,
    meanest_ibes, stdev_ibes, numest_ibes, dispersion_pct_ibes,
    momentum_1m, momentum_3m, momentum_6m, rolling_momentum_3m, z_score_momentum
"""

import os
import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay
import wrds
import warnings
warnings.filterwarnings('ignore')

# ---------------- Config ----------------
UNIVERSE_CSV   = os.getenv("UNIVERSE_CSV", "analysis_scripts_2/data_files/top500_liquidity_2005_2023.csv")
EARNINGS_CSV   = os.getenv("EARNINGS_CSV", "analysis_scripts_2/data_files/earnings_dates.csv")
OUT_DIR        = os.getenv("OUT_DIR", "analysis_scripts_2")
START_DATE     = os.getenv("START_DATE", "2000-01-01")
END_DATE       = os.getenv("END_DATE", "2025-12-31")
SAMPLE_CUSIPS  = os.getenv("SAMPLE_CUSIPS", "")  # e.g. "AAPL=03783310,MSFT=59491810" (uses CUSIPs only)

os.makedirs(os.path.join(OUT_DIR, "data_files"), exist_ok=True)

# --------------- Helpers ----------------
def _normalize_cusip_8(series: pd.Series) -> pd.Series:
    s = series.fillna("").astype(str).str.strip()
    s = s.str.replace(r"[^A-Za-z0-9]", "", regex=True)
    return s.str[:8]

def load_universe(universe_csv: str) -> pd.DataFrame:
    """Return unique cusip8,ticker from universe file if present."""
    try:
        df = pd.read_csv(universe_csv)
    except Exception:
        return pd.DataFrame(columns=["cusip8","ticker"])
    if "cusip" in df.columns:
        df["cusip8"] = _normalize_cusip_8(df["cusip"])
    elif "cusip8" in df.columns:
        df["cusip8"] = _normalize_cusip_8(df["cusip8"])
    else:
        df["cusip8"] = np.nan
    if "ticker" not in df.columns:
        df["ticker"] = np.nan
    df["ticker"] = df["ticker"].astype(str).str.strip()
    return df[["cusip8","ticker"]].drop_duplicates().reset_index(drop=True)

def fetch_ibes_estimates_by_ids(db, cusip8_list: list, ticker_list: list,
                                start_date: str, end_date: str) -> pd.DataFrame:
    """
    Query tr_ibes.statsum_epsus with OR on cusip or IBES ticker, in manageable batches.
    """
    cusips = sorted({c for c in cusip8_list if isinstance(c, str) and c})
    tickers = sorted({t for t in ticker_list if isinstance(t, str) and t})

    results = []
    C_BATCH, T_BATCH = 800, 800
    c_batches = [cusips[i:i+C_BATCH] for i in range(0, max(len(cusips), 1), C_BATCH)] or [[]]
    t_batches = [tickers[i:i+T_BATCH] for i in range(0, max(len(tickers), 1), T_BATCH)] or [[]]

    for cb in c_batches:
        for tb in t_batches:
            clauses = []
            if cb:
                clauses.append("cusip IN (" + ", ".join(f"'{c}'" for c in cb) + ")")
            if tb:
                clauses.append("ticker IN (" + ", ".join(f"'{t}'" for t in tb) + ")")
            if not clauses:
                continue
            where_ids = " OR ".join(clauses)
            q = f"""
                SELECT ticker, cusip, statpers, fpedats, anndats_act,
                       meanest, stdev, numest, fpi, measure, fiscalp
                FROM tr_ibes.statsum_epsus
                WHERE ({where_ids})
                  AND statpers BETWEEN '{start_date}' AND '{end_date}'
                  AND measure = 'EPS'
                  AND fiscalp = 'QTR'
                  AND meanest IS NOT NULL
            """
            part = db.raw_sql(q, date_cols=["statpers","fpedats","anndats_act"])
            if not part.empty:
                results.append(part)

    if not results:
        return pd.DataFrame(columns=[
            "ticker","cusip","statpers","fpedats","anndats_act",
            "meanest","stdev","numest","fpi","measure","fiscalp"
        ])
    return pd.concat(results, ignore_index=True)

def compute_momentum(ts: pd.Series) -> pd.DataFrame:
    """Return only momentum features (no 'meanest' to avoid join overlaps)."""
    s = ts.sort_index()
    out = pd.DataFrame(index=s.index)
    out["momentum_1m"] = s.pct_change(20)
    out["momentum_3m"] = s.pct_change(60)
    out["momentum_6m"] = s.pct_change(120)
    out["rolling_momentum_3m"] = s.rolling(60, min_periods=60).mean().pct_change(20)
    roll_mean_1y = s.rolling(252, min_periods=60).mean()
    roll_std_1y  = s.rolling(252, min_periods=60).std()
    out["z_score_momentum"] = (s - roll_mean_1y) / roll_std_1y
    out["z_score_momentum_smoothed"] = (out["z_score_momentum"].rolling(60, min_periods=20).mean())
    return out

# --------------- Main -------------------
def main():
    print("Building EPS features at analysis dates (earnings_date - 21 BDays)…")

    # Earnings schedule
    earn = pd.read_csv(EARNINGS_CSV)
    if "earnings_date" not in earn.columns:
        raise ValueError("earnings_dates.csv must have an 'earnings_date' column")
    earn["earnings_date"] = pd.to_datetime(earn["earnings_date"], errors="coerce")

    # Normalize IDs from earnings file
    if "cusip" in earn.columns:
        earn["cusip8"] = _normalize_cusip_8(earn["cusip"])
    elif "cusip8" in earn.columns:
        earn["cusip8"] = _normalize_cusip_8(earn["cusip8"])
    else:
        earn["cusip8"] = np.nan
    if "ticker" in earn.columns:
        earn["ticker"] = earn["ticker"].astype(str).str.strip()
    else:
        earn["ticker"] = np.nan

    # analysis_date = T - 21BD
    earn["analysis_date"] = earn["earnings_date"] - BDay(21)

    # IBES window (add lookback for momentum)
    min_needed = earn["analysis_date"].min() - BDay(260)
    max_needed = earn["analysis_date"].max()
    start_date = str(min(min_needed, pd.to_datetime(START_DATE)).date())
    end_date   = str(max(max_needed, pd.to_datetime(END_DATE)).date())
    print(f"IBES fetch window: {start_date} .. {end_date}")

    # WRDS connection
    try:
        db = wrds.Connection()
        print("✓ Connected to WRDS")
    except Exception as e:
        raise RuntimeError(f"WRDS connection failed: {e}")

    # Universe fallback
    uni = load_universe(UNIVERSE_CSV)

    # Build identifier sets
    cusips_from_earn  = earn["cusip8"].dropna().tolist()
    tickers_from_earn = earn["ticker"].dropna().tolist()
    cusips_from_uni   = uni["cusip8"].dropna().tolist() if not uni.empty else []
    tickers_from_uni  = uni["ticker"].dropna().tolist()  if not uni.empty else []

    cusip8_list = sorted(set(cusips_from_earn) | set(cusips_from_uni))
    ticker_list = sorted(set(tickers_from_earn) | set(tickers_from_uni))

    if SAMPLE_CUSIPS:
        cusip8_list = [t.split("=")[-1][:8] for t in SAMPLE_CUSIPS.split(",") if t.strip()]
        print(f"Using SAMPLE_CUSIPS override: {cusip8_list}")

    print(f"IBES identifier sets -> cusips: {len(cusip8_list)}, tickers: {len(ticker_list)}")

    # IBES fetch using BOTH lists (CUSIP OR TICKER)
    ibes = fetch_ibes_estimates_by_ids(db, cusip8_list, ticker_list, start_date, end_date)
    if ibes.empty:
        raise RuntimeError("No IBES data returned for selected identifiers/window.")

    # Sanity log
    print("Sample IBES rows:")
    print(ibes[["ticker","cusip","statpers","meanest","stdev","numest"]].head(10).to_string(index=False))

    # Parse dates
    ibes["statpers"] = pd.to_datetime(ibes["statpers"])
    ibes["fpedats"]  = pd.to_datetime(ibes["fpedats"])

    # One-quarter-ahead proxy: choose first fpedats > statpers per (cusip,ticker,statpers)
    ibes = (
        ibes.loc[ibes["fpedats"] > ibes["statpers"]]
            .sort_values(["cusip","ticker","statpers","fpedats"])
            .groupby(["cusip","ticker","statpers"], as_index=False)
            .first()
    )

    # Build per-(cusip,ticker) time series features
    feats_frames = []
    for (cusip, ticker), g in ibes.groupby(["cusip","ticker"]):
        g = g.sort_values("statpers").set_index("statpers")
        feats = compute_momentum(g["meanest"])  # momentum only
        # join without overlapping 'meanest'
        joined = g[["meanest","stdev","numest"]].join(feats, how="left")
        joined["dispersion_pct"] = (joined["stdev"].abs() / joined["meanest"].abs()).replace([np.inf, -np.inf], np.nan)
        joined["cusip"] = cusip
        joined["cusip8"] = str(cusip)[:8] if isinstance(cusip, str) else np.nan
        joined["ticker"] = str(ticker) if isinstance(ticker, str) else np.nan
        feats_frames.append(joined.reset_index().rename(columns={"statpers":"date"}))

    features = pd.concat(feats_frames, ignore_index=True) if feats_frames else pd.DataFrame()
    if features.empty:
        raise RuntimeError("No features computed — check identifier sets and IBES data.")

    # As-of merge keys: prefer cusip8 if present in earnings, else ticker
    group_by_cusip = earn["cusip8"].notna().any()
    key_cols = ["cusip8"] if group_by_cusip else ["ticker"]

    # As-of merge with suffixes; keep IDs from both sides and coalesce
    out_frames = []
    for key_vals, sub in earn.groupby(key_cols):
        if not isinstance(key_vals, tuple):
            key_vals = (key_vals,)
        # filter features by key
        mask = np.ones(len(features), dtype=bool)
        for k, v in zip(key_cols, key_vals):
            mask &= (features[k] == v)
        fsub = features.loc[mask].sort_values("date")
        if fsub.empty:
            continue

        # left: ensure IDs carried through
        left_cols = [c for c in ["cusip8","ticker","earnings_date","analysis_date"] if c in sub.columns]
        esub = sub[left_cols].sort_values("analysis_date").copy()

        # right: metrics & IDs (rename to *_ibes)
        right_cols = [
            "date","cusip8","ticker",
            "meanest","stdev","numest","dispersion_pct",
            "momentum_1m","momentum_3m","momentum_6m",
            "rolling_momentum_3m","z_score_momentum_smoothed"
        ]
        right_cols = [c for c in right_cols if c in fsub.columns]
        fsub_use = fsub[right_cols].rename(columns={
            "meanest": "meanest_ibes",
            "stdev": "stdev_ibes",
            "numest": "numest_ibes",
            "dispersion_pct": "dispersion_pct_ibes",
            "z_score_momentum_smoothed": "z_score_momentum",
        })

        merged = pd.merge_asof(
            esub,
            fsub_use,
            left_on="analysis_date",
            right_on="date",
            direction="backward",
            suffixes=("_earn","_ibes")
        )

        # Coalesce IDs: prefer earn side, then IBES side
        if "cusip8_earn" in merged.columns or "cusip8_ibes" in merged.columns:
            merged["cusip8"] = merged.get("cusip8_earn", pd.Series(index=merged.index)).fillna(
                               merged.get("cusip8_ibes", np.nan))
        if "ticker_earn" in merged.columns or "ticker_ibes" in merged.columns:
            merged["ticker"] = merged.get("ticker_earn", pd.Series(index=merged.index)).fillna(
                               merged.get("ticker_ibes", np.nan))

        out_frames.append(merged)

    long_table = pd.concat(out_frames, ignore_index=True) if out_frames else pd.DataFrame()

    # Final column order
    keep_cols = [
        "cusip8","ticker",
        "earnings_date","analysis_date","date",  # matched statpers
        "meanest_ibes","stdev_ibes","numest_ibes","dispersion_pct_ibes",
        "momentum_1m","momentum_3m","momentum_6m",
        "rolling_momentum_3m","z_score_momentum",
    ]
    existing = [c for c in keep_cols if c in long_table.columns]
    long_table = long_table[existing].rename(columns={"date":"matched_statpers"})

    out_path = os.path.join(OUT_DIR, "data_files", "eps_features_at_analysis_dates.csv")
    long_table.to_csv(out_path, index=False)

    print("\n=== Done ===")
    print(f"✓ Wrote: {out_path}")
    print(f"Rows: {len(long_table)} | Columns: {len(long_table.columns)}")

if __name__ == "__main__":
    main()

