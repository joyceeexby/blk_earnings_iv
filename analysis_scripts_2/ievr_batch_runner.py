#!/usr/bin/env python3
"""
Robust, batched IEVR runner

- Loads local CSVs: top500 universe, CCM link table, earnings_dates
- Builds (ticker, earnings_date, analysis_date) jobs per quarter
- Resolves OptionMetrics secid using historical symbol table (secnmd) as-of analysis date
- Pulls IV surfaces with caching, tries both opprcdYYYY and opprcd
- Widened filters and hardened kernel regression for sparse old data
- Saves results to analysis_scripts_2/data_files/ievr_batch_YYYYMMDD_HHMMSS.csv
"""

import os
import sys
import math
import argparse
from datetime import datetime
import pandas as pd
import numpy as np

# --- Paths ---
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data_files")
UNIVERSE_CSV = os.path.join(DATA_DIR, "top500_liquidity_2005_2023.csv")
LINK_CSV     = os.path.join(DATA_DIR, "link_table.csv")
EARN_CSV     = os.path.join(DATA_DIR, "earnings_dates.csv")

# Import your IEVR implementation
sys.path.append(HERE)  # so we can import ievr_analysis.py beside this file
from ievr_analysis import IEVRAnalysis

# ------------------------------------------------------------------------------------
# Small caching layer for IV surfaces to avoid repeated WRDS pulls
# ------------------------------------------------------------------------------------

class IVSurfaceCache:
    def __init__(self):
        self._cache = {}  # (ticker, closest_date_str) -> DataFrame

    def get(self, ticker, closest_date_str):
        return self._cache.get((ticker, closest_date_str))

    def put(self, ticker, closest_date_str, iv_df):
        self._cache[(ticker, closest_date_str)] = iv_df

# ------------------------------------------------------------------------------------
# Robust historical secid lookup (symbol history as-of the analysis date)
# ------------------------------------------------------------------------------------
def get_secid_at_date(db, ticker, analysis_date):
    """
    Resolve OptionMetrics secid for a given ticker as of analysis_date
    using optionm.secnmd (no end_date in this table).
    Strategy: pick the most recent row with effect_date <= analysis_date.
    """
    asof = pd.to_datetime(analysis_date).strftime("%Y-%m-%d")
    q = f"""
        SELECT secid, ticker, effect_date
        FROM optionm.secnmd
        WHERE UPPER(ticker) = UPPER('{ticker}')
          AND effect_date <= '{asof}'
        ORDER BY effect_date DESC
        LIMIT 1
    """
    df = db.raw_sql(q, date_cols=['effect_date'])
    if df.empty:
        return None
    return int(df.iloc[0]['secid'])



# def get_secid_at_date(db, ticker, asof_date):
#     """
#     Resolve OptionMetrics secid for a given ticker as of a specific date,
#     using the symbol name history table (secnmd). Returns int or None.
#     """
#     asof = pd.to_datetime(asof_date).strftime("%Y-%m-%d")
#     q = f"""
#         SELECT secid, ticker, effect_date, COALESCE(name_end, DATE '9999-12-31') AS end_date
#         FROM optionm.secnmd
#         WHERE UPPER(ticker) = UPPER('{ticker}')
#         AND effect_date <= '{analysis_date}'
#         AND COALESCE(name_end, DATE '9999-12-31') >= '{analysis_date}'
#         ORDER BY effect_date DESC
#         LIMIT 1
#     """
#     df = db.raw_sql(q, date_cols=['effect_date','end_date'])
#     if df.empty:
#         # Fallback: most recent symbol record <= asof
#         q2 = f"""
#             SELECT secid, ticker, effect_date, COALESCE(end_date, DATE '9999-12-31') AS end_date
#             FROM optionm.secnmd
#             WHERE UPPER(ticker) = UPPER('{ticker}')
#               AND effect_date <= '{asof}'
#             ORDER BY effect_date DESC
#             LIMIT 1
#         """
#         df = db.raw_sql(q2, date_cols=['effect_date','end_date'])
#         if df.empty:
#             return None
#     return int(df.iloc[0]['secid'])

# ------------------------------------------------------------------------------------
# Fetch IV surface around analysis date, try both opprcdYYYY/opprcd, cache by (ticker, date)
# ------------------------------------------------------------------------------------

def fetch_iv_surface_for_date(db, ticker, analysis_date, cache, window_days=30):
    """
    Fetch the IV rows for the closest trading date to `analysis_date` (±window_days),
    cache the per-day cleaned surface for reuse.
    """
    secid = get_secid_at_date(db, ticker, analysis_date)
    if secid is None:
        print(f"[WARN] No secid (symbol history) for {ticker} @ {analysis_date:%Y-%m-%d}")
        return None, None

    start_date = (pd.to_datetime(analysis_date) - pd.Timedelta(days=window_days)).strftime("%Y-%m-%d")
    end_date   = (pd.to_datetime(analysis_date) + pd.Timedelta(days=window_days)).strftime("%Y-%m-%d")

    # Discover candidate option price tables
    year = pd.to_datetime(analysis_date).year
    tables = db.raw_sql("""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'optionm' AND table_name LIKE 'opprcd%%'
    """)
    avail = set(tables['table_name'].str.lower())
    candidates = []
    if f"opprcd{year}" in avail:
        candidates.append(f"opprcd{year}")
    if "opprcd" in avail:
        candidates.append("opprcd")

    iv_all = pd.DataFrame()
    for tbl in candidates:
        q_iv = f"""
            SELECT date, exdate, strike_price, cp_flag, impl_volatility
            FROM optionm.{tbl}
            WHERE secid = {secid}
              AND date BETWEEN '{start_date}' AND '{end_date}'
              AND impl_volatility > 0 AND impl_volatility < 5.0
            ORDER BY date, exdate, strike_price
        """
        try:
            part = db.raw_sql(q_iv)
        except Exception as e:
            print(f"[WARN] Skipping {tbl}: {e}")
            continue
        if not part.empty:
            iv_all = pd.concat([iv_all, part], ignore_index=True)

    if iv_all.empty:
        print(f"[WARN] No IV rows for {ticker} in window {start_date}..{end_date}")
        return None, None

    iv_all['date'] = pd.to_datetime(iv_all['date'])
    iv_all['exdate'] = pd.to_datetime(iv_all['exdate'])

    # Choose closest trading date
    ad = pd.to_datetime(analysis_date)
    closest_idx = (iv_all['date'] - ad).abs().idxmin()
    closest_date = iv_all.loc[closest_idx, 'date'].normalize()
    closest_date_str = closest_date.strftime("%Y-%m-%d")

    # Return cache if exists
    cached = cache.get(ticker, closest_date_str)
    if cached is not None:
        return cached.copy(), closest_date

    # Build per-day surface and cache it
    iv_day = iv_all[iv_all['date'] == closest_date].copy()
    if iv_day.empty:
        return None, None

    # Underlying
    q_px = f"""
        SELECT close
        FROM optionm.secprd
        WHERE secid = {secid} AND date = '{closest_date_str}'
    """
    px = db.raw_sql(q_px)
    underlying = float(px.iloc[0]['close']) if not px.empty else 100.0

    # Compute moneyness and TTE
    iv_day['underlying_price'] = underlying
    iv_day['moneyness'] = (iv_day['strike_price'] / 1000.0) / underlying
    iv_day['tte'] = (iv_day['exdate'] - iv_day['date']).dt.days

    # Widened filters for sparse old data
    iv_day = iv_day[
        (iv_day['moneyness'].between(0.95, 1.05)) &
        (iv_day['tte'].between(7, 120))
    ].copy()

    puts = iv_day[iv_day['cp_flag'] == 'P'].copy()
    calls = iv_day[iv_day['cp_flag'] == 'C'].copy()

    final = puts[['tte', 'moneyness', 'impl_volatility', 'strike_price', 'underlying_price', 'cp_flag']].copy()
    final.columns = ['tte', 'moneyness', 'put_iv', 'strike_price', 'underlying_price', 'cp_flag']
    if not calls.empty:
        c = calls[['tte', 'moneyness', 'impl_volatility']].copy()
        c.columns = ['tte', 'moneyness', 'call_iv']
        final = final.merge(c, on=['tte', 'moneyness'], how='left')
    else:
        final['call_iv'] = final['put_iv']

    cache.put(ticker, closest_date_str, final.copy())
    return final, closest_date

# ------------------------------------------------------------------------------------
# Build batch jobs from local CSVs
# ------------------------------------------------------------------------------------

def build_jobs_from_csvs(universe_csv, link_csv, earnings_csv, analysis_days_before=30):
    """
    Returns DataFrame with:
      ['ticker','gvkey','permno','quarter_start_date','earnings_date','analysis_date']
    """
    top500 = pd.read_csv(universe_csv, parse_dates=['quarter_start_date', 'quarter_end_date'])
    links  = pd.read_csv(link_csv,     parse_dates=['linkdt','linkenddt'])
    earns  = pd.read_csv(earnings_csv, parse_dates=['earnings_date','datadate'])

    links['linkenddt'] = links['linkenddt'].fillna(pd.Timestamp('2099-12-31'))

    req_univ_cols = {'permno','quarter_start_date','quarter_end_date'}
    if not req_univ_cols.issubset(set(top500.columns)):
        raise ValueError(f"Universe file missing required columns {req_univ_cols}. Got: {top500.columns.tolist()}")

    # Keep "good" links
    good = links.query("linktype in ['LU','LC'] and linkprim in ['P','C']").copy()

    tmp = top500.merge(
        good,
        left_on='permno',
        right_on='lpermno',
        how='left',
        suffixes=('','_link')
    )

    # Quarter start must be within link validity window
    tmp = tmp[(tmp['quarter_start_date'] >= tmp['linkdt']) & (tmp['quarter_start_date'] <= tmp['linkenddt'])].copy()

    # Earnings by gvkey; NOTE: earnings csv must have 'ticker' column (not 'tic')
    if 'ticker' not in earns.columns:
        raise ValueError("earnings_dates.csv must contain a column named 'ticker' (was previously 'tic').")

    merged = tmp.merge(
        earns[['gvkey','earnings_date','ticker']].rename(columns={'ticker':'ticker_from_earn'}),
        on='gvkey',
        how='left'
    )

    # Keep earnings inside the quarter window
    merged = merged[
        (merged['earnings_date'] >= merged['quarter_start_date']) &
        (merged['earnings_date'] <= merged['quarter_end_date'])
    ].copy()

    # Prefer earnings ticker; fall back to universe ticker if you have one
    if 'ticker' in merged.columns and merged['ticker'].notna().any():
        merged['ticker_final'] = merged['ticker_from_earn'].fillna(merged['ticker'])
    else:
        merged['ticker_final'] = merged['ticker_from_earn']

    merged = merged[merged['ticker_final'].notna()].copy()

    merged['analysis_date'] = merged['earnings_date'] - pd.to_timedelta(analysis_days_before, unit='D')

    jobs = merged[['ticker_final','gvkey','permno','quarter_start_date','earnings_date','analysis_date']].copy()
    jobs = jobs.rename(columns={'ticker_final':'ticker'})
    jobs = jobs.drop_duplicates(subset=['ticker','earnings_date']).reset_index(drop=True)
    return jobs

# ------------------------------------------------------------------------------------
# Batch runner
# ------------------------------------------------------------------------------------

def run_batch_ievr(jobs_df, analysis_days_before, out_csv_path, wrds_user=None, window_days=30, post_extra_days=30):
    import wrds
    db = wrds.Connection(wrds_username=wrds_user) if wrds_user else wrds.Connection()

    cache = IVSurfaceCache()
    analyzer = IEVRAnalysis(db)

    results = []

    def kernel_avg_iv(df):
        df = df.dropna(subset=['tte','put_iv'])
        if df.empty:
            return float('nan')
        tte = df['tte'].values
        ivs = df['put_iv'].values
        if len(df) == 1:
            return float(ivs[0])  # nearest-neighbor fallback
        # grid size bounded by number of points
        tte_grid = np.linspace(df['tte'].min(), df['tte'].max(), max(5, min(10, len(df))))
        vals = [analyzer.kernel_regression_iv(tte, ivs, t) for t in tte_grid]
        vals = [v for v in vals if np.isfinite(v)]
        if not vals:
            return float('nan')
        return float(np.mean(vals))

    for i, row in jobs_df.iterrows():
        ticker = row['ticker']
        edate  = pd.to_datetime(row['earnings_date'])
        adate  = pd.to_datetime(row['analysis_date'])

        print(f"\n[{i+1}/{len(jobs_df)}] {ticker} | earnings {edate.date()} | analysis {adate.date()}")

        iv_surface, closest_date = fetch_iv_surface_for_date(db, ticker, adate, cache, window_days=window_days)
        if iv_surface is None:
            print(f"  -> Skipping {ticker}: no IV surface.")
            continue

        analyzer.earnings_date = edate
        analyzer.analysis_date = adate

        days_to_earnings = (edate - adate).days
        atm = iv_surface[iv_surface['moneyness'].between(0.95, 1.05)]

        pre  = atm[atm['tte'] <  days_to_earnings].copy()
        post = atm[(atm['tte'] > days_to_earnings) & (atm['tte'] <= days_to_earnings + post_extra_days)].copy()

        avg_pre  = kernel_avg_iv(pre)
        avg_post = kernel_avg_iv(post)

        if (not np.isfinite(avg_pre)) or (not np.isfinite(avg_post)) or avg_pre == 0:
            print("  -> Not enough ATM data; skipping.")
            continue

        ievr = float(avg_post / avg_pre)

        underlying = iv_surface['underlying_price'].iloc[0] if 'underlying_price' in iv_surface.columns else 100.0
        skew = analyzer.calculate_skew_ratio(iv_surface, underlying)

        results.append({
            'ticker': ticker,
            'earnings_date': edate.date(),
            'analysis_date': adate.date(),
            'closest_quote_date': closest_date.date(),
            'days_to_earnings': days_to_earnings,
            'ievr': ievr,
            'avg_pre': float(avg_pre),
            'avg_post': float(avg_post),
            'skew_ratio': float(skew) if np.isfinite(skew) else np.nan
        })

    db.close()

    if not results:
        print("No results computed.")
        return pd.DataFrame()

    out = pd.DataFrame(results).sort_values(['analysis_date','ticker']).reset_index(drop=True)
    os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)
    out.to_csv(out_csv_path, index=False)
    print(f"\n✓ Saved {len(out):,} rows to {out_csv_path}")
    return out

# ------------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------------

def auto_outfile(prefix="ievr_batch"):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(DATA_DIR, f"{prefix}_{ts}.csv")

def main():
    parser = argparse.ArgumentParser(description="Vectorized/batched IEVR runner (robust)")
    parser.add_argument("--analysis-days-before", type=int, default=30, help="Days before earnings (T - x)")
    parser.add_argument("--wrds-user", type=str, default=None, help="WRDS username (optional)")
    parser.add_argument("--outfile", type=str, default=None, help="Output CSV path")
    parser.add_argument("--window-days", type=int, default=30, help="IV fetch window (±days) around analysis date")
    parser.add_argument("--post-extra-days", type=int, default=30, help="Post window length beyond event")
    args = parser.parse_args()

    jobs = build_jobs_from_csvs(UNIVERSE_CSV, LINK_CSV, EARN_CSV, analysis_days_before=args.analysis_days_before)
    print(f"Built {len(jobs):,} jobs.")

    outfile = args.outfile or auto_outfile()
    run_batch_ievr(
        jobs,
        analysis_days_before=args.analysis_days_before,
        out_csv_path=outfile,
        wrds_user=args.wrds_user,
        window_days=args.window_days,
        post_extra_days=args.post_extra_days
    )

if __name__ == "__main__":
    main()
