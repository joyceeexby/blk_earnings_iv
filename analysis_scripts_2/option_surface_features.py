
# This file contains functions for computing option surface features.

import pandas as pd
import numpy as np
import wrds
from datetime import datetime

def extract_skew_feature(secid, earnings_date, db, n_lag=20):
    query_date = get_relative_surface_date(secid, earnings_date, n_lag, db)
    if query_date is None:
        return None, None

    df = get_surface_row(secid, query_date, db)
    df = df[df['days'] == 30]

    if df.empty:
        return None, query_date

    call_otm = df[(df['cp_flag'] == 'C') & (df['delta'] == 25.0)]
    put_otm = df[(df['cp_flag'] == 'P') & (df['delta'] == -25.0)]
    call_atm = df[(df['cp_flag'] == 'C') & (df['delta'] == 50.0)]
    put_atm = df[(df['cp_flag'] == 'P') & (df['delta'] == -50.0)]

    if call_otm.empty or put_otm.empty or call_atm.empty or put_atm.empty:
        return None, query_date

    atm_iv = (call_atm['impl_volatility'].mean() + put_atm['impl_volatility'].mean()) / 2
    skew = (call_otm['impl_volatility'].mean() - put_otm['impl_volatility'].mean()) / atm_iv
    return skew, query_date

def get_latest_ticker_info(ticker_list, db):
    """
    Fetches the latest effect_date, secid, and ticker from optionm_all.secnmd table
    for the given list of tickers, keeping only the latest secid per ticker.

    Parameters:
    - ticker_list (list of str): List of ticker symbols to query.
    - db: A WRDS database connection object with a `.raw_sql()` method.

    Returns:
    - pandas.DataFrame: DataFrame containing ticker, secid, and the latest effect_date.
    """
    if not ticker_list:
        return pd.DataFrame(columns=["ticker", "secid", "effect_date"])

    # Format tickers safely
    formatted_tickers = "', '".join(ticker_list)

    # Fetch all secids and their effect dates for the given tickers
    query_ticker = f"""
    SELECT ticker, secid, effect_date
    FROM optionm_all.secnmd
    WHERE ticker IN ('{formatted_tickers}')
    ORDER BY ticker, effect_date DESC;
    """

    df = db.raw_sql(query_ticker)

    # Keep only the row with the latest effect_date for each ticker
    # Sort by ticker and effect_date descending, then group by ticker and take the first row
    df = df.sort_values(by=['ticker', 'effect_date'], ascending=[True, False])
    df = df.groupby('ticker').head(1).reset_index(drop=True)

    return df

def get_surface_row(secid, query_date, db):
    year = pd.to_datetime(query_date).year
    table_name = f"optionm_all.vsurfd{year}"

    query = f"""
    SELECT *
    FROM {table_name}
    WHERE secid = {secid}
      AND date = '{query_date}'
      AND days BETWEEN 7 AND 60
    """
    df = db.raw_sql(query)
    return df



def extract_kurtosis_feature(secid, earnings_date, db, n_lag=20):
    query_date = get_relative_surface_date(secid, earnings_date, n_lag, db)
    if query_date is None:
        return None, None

    df = get_surface_row(secid, query_date, db)
    df = df[df['days'] == 30]

    if df.empty:
        return None, query_date

    call_otm = df[(df['cp_flag'] == 'C') & (df['delta'] == 25.0)]
    put_otm = df[(df['cp_flag'] == 'P') & (df['delta'] == -25.0)]
    call_atm = df[(df['cp_flag'] == 'C') & (df['delta'] == 50.0)]
    put_atm = df[(df['cp_flag'] == 'P') & (df['delta'] == -50.0)]

    if call_otm.empty or put_otm.empty or call_atm.empty or put_atm.empty:
        return None, query_date

    atm_iv = (call_atm['impl_volatility'].mean() + put_atm['impl_volatility'].mean()) / 2
    kurtosis = (
        (call_otm['impl_volatility'].mean() + put_otm['impl_volatility'].mean()) -
        (call_atm['impl_volatility'].mean() + put_atm['impl_volatility'].mean())
    ) / atm_iv

    return kurtosis, query_date

def compute_option_surface_features(ticker_list, earnings_year, earnings_quarter, db, n_lag=20):
    """
    Compute option surface features for a list of tickers and a given earnings season.

    Parameters:
    - ticker_list (list[str]): List of stock tickers to include
    - earnings_year (int): Earnings year (e.g., 2021)
    - earnings_quarter (int): Earnings quarter (1, 2, 3, 4)
    - db: WRDS database connection
    - n_lag (int): Trading days before earnings to measure features

    Returns:
    - pandas.DataFrame: DataFrame with calculated features
    """
    print("Starting compute_option_surface_features...")
    # Define start and end dates based on earnings year and quarter
    start_earnings_date = f"{earnings_year - 1}-09-30"
    end_earnings_date = f"{earnings_year + 1}-03-30"
    print(f"Fetching earnings data between {start_earnings_date} and {end_earnings_date}")

    # Get earnings data internally
    earnings_df = get_earnings_data(ticker_list, start_earnings_date, end_earnings_date, db)
    print(f"Fetched {len(earnings_df)} earnings records.")

    # Filter earnings data by actual earnings date falling within the calendar quarter
    # Define quarter date ranges
    quarter_start_dates = {
        1: f"{earnings_year}-01-01",
        2: f"{earnings_year}-04-01", 
        3: f"{earnings_year}-07-01",
        4: f"{earnings_year}-10-01"
    }
    quarter_end_dates = {
        1: f"{earnings_year}-03-31",
        2: f"{earnings_year}-06-30",
        3: f"{earnings_year}-09-30", 
        4: f"{earnings_year}-12-31"
    }
    
    start_date = quarter_start_dates[earnings_quarter]
    end_date = quarter_end_dates[earnings_quarter]
    
    # Filter by actual earnings date within the calendar quarter
    earnings_df = earnings_df[
        (earnings_df['earnings_date'] >= start_date) &
        (earnings_df['earnings_date'] <= end_date)
    ].copy()
    print(f"Filtered down to {len(earnings_df)} earnings records with dates in {start_date} to {end_date}.")

    # Map tickers to secids
    print("Mapping tickers to secids...")
    secid_map = get_latest_ticker_info(earnings_df['ticker'].to_list(), db)
    print(f"Mapped {len(secid_map)} tickers to secids.")

    # Merge secid into earnings data
    earnings_df = earnings_df.merge(secid_map, on="ticker", how="left").dropna(subset=["secid"])
    print(f"Merged secids. Remaining earnings records: {len(earnings_df)}")

    results = []
    total_tickers = len(earnings_df)
    print(f"Starting feature extraction for {total_tickers} tickers...")

    for i, row in earnings_df.iterrows():
        ticker = row['ticker']
        secid = int(row['secid'])
        earnings_date = row['earnings_date']

        print(f"Processing ticker {i+1}/{total_tickers}: {ticker} (SECID: {secid}, Earnings Date: {earnings_date})")

        try:
            surface_date = get_relative_surface_date(secid, earnings_date, n_lag, db)
            print(f"  - Retrieved surface date: {surface_date}")
        except Exception as e:
            surface_date = None
            print(f"  - Error getting surface date: {e}")

        try:
            term_ratio, _ = extract_term_diff_feature(secid, earnings_date, db, n_lag)
            print(f"  - Computed TERM_RATIO: {term_ratio}")
        except Exception as e:
            term_ratio = None
            print(f"  - Error computing TERM_RATIO: {e}")

        try:
            skew, _ = extract_skew_feature(secid, earnings_date, db, n_lag)
            print(f"  - Computed SKEW: {skew}")
        except Exception as e:
            skew = None
            print(f"  - Error computing SKEW: {e}")

        try:
            kurt, _ = extract_kurtosis_feature(secid, earnings_date, db, n_lag)
            print(f"  - Computed KURT: {kurt}")
        except Exception as e:
            kurt = None
            print(f"  - Error computing KURT: {e}")

        try:
            iv_ratio, iv_recent_date, iv_earlier_date = monthly_iv_change_ratio_feature(secid, earnings_date, db, n_lag)
            print(f"  - Computed IV_RATIO: {iv_ratio} (Dates: {iv_recent_date}, {iv_earlier_date})")
        except Exception as e:
            iv_ratio = None
            print(f"  - Error computing IV_RATIO: {e}")

        try:
            smirk = extract_smirk_feature(secid, earnings_date, db, n_lag)
            if isinstance(smirk, tuple):
                smirk = smirk[0]
            print(f"  - Computed SMIRK: {smirk}")
        except Exception as e:
            smirk = None
            print(f"  - Error computing SMIRK: {e}")

        results.append({
            'ticker': ticker,
            'secid': secid,
            'earnings_date': earnings_date,
            'surface_date': surface_date,
            'TERM_RATIO': term_ratio,
            'SKEW': skew,
            'KURT': kurt,
            'IV_RATIO': iv_ratio,
            'SMIRK': smirk
        })
        print("-" * 20)  # Separator for clarity

    print("Feature extraction complete.")
    return pd.DataFrame(results)

def monthly_iv_change_ratio_feature(secid, earnings_date, db, n_lag=20, monthly_lag=21):
    """
    Computes the ratio of ATM implied volatility between t-n_lag and t-(n_lag + monthly_lag),
    where t is the earnings_date. Uses 30-day options.

    Parameters:
    - secid (int): WRDS OptionMetrics security ID.
    - earnings_date (str): Earnings announcement date in 'YYYY-MM-DD' format.
    - db: WRDS database connection.
    - n_lag (int): Days before earnings_date to compute latest IV (default=20).
    - monthly_lag (int): Additional trading days back for comparison (default=21).

    Returns:
    - iv_ratio (float): Ratio of ATM IV at t-n_lag over t-(n_lag + monthly_lag).
    - query_date (str): Trade date for t-n_lag.
    - earlier_date (str): Trade date for t-(n_lag + monthly_lag).
    """

    query_date = get_relative_surface_date(secid, earnings_date, n_lag, db)
    earlier_date = get_relative_surface_date(secid, earnings_date, n_lag + monthly_lag, db)

    if query_date is None or earlier_date is None:
        return None, query_date, earlier_date

    df_recent = get_surface_row(secid, query_date, db)
    df_earlier = get_surface_row(secid, earlier_date, db)

    df_recent = df_recent[df_recent['days'] == 30]
    df_earlier = df_earlier[df_earlier['days'] == 30]

    if df_recent.empty or df_earlier.empty:
        return None, query_date, earlier_date

    call_atm_recent = df_recent[(df_recent['cp_flag'] == 'C') & (df_recent['delta'] == 50.0)]
    put_atm_recent = df_recent[(df_recent['cp_flag'] == 'P') & (df_recent['delta'] == -50.0)]
    call_atm_earlier = df_earlier[(df_earlier['cp_flag'] == 'C') & (df_earlier['delta'] == 50.0)]
    put_atm_earlier = df_earlier[(df_earlier['cp_flag'] == 'P') & (df_earlier['delta'] == -50.0)]

    if call_atm_recent.empty or put_atm_recent.empty or call_atm_earlier.empty or put_atm_earlier.empty:
        return None, query_date, earlier_date

    iv_recent = (call_atm_recent['impl_volatility'].mean() + put_atm_recent['impl_volatility'].mean()) / 2
    iv_earlier = (call_atm_earlier['impl_volatility'].mean() + put_atm_earlier['impl_volatility'].mean()) / 2

    if iv_earlier == 0:
        return None, query_date, earlier_date

    iv_ratio = iv_recent / iv_earlier
    return iv_ratio, query_date, earlier_date

def get_relative_surface_date(secid, earnings_date, n_lag, db):
    """
    Get the trading date that is `n_lag` trading days before the earnings date.
    Only uses available dates in optionm_all.vsurfd tables for this secid.

    Returns:
        query_date (str) or None
    """
    edate = pd.to_datetime(earnings_date)
    years = {edate.year, edate.year - 1}
    surface_dates = []

    for year in years:
        table = f"optionm_all.vsurfd{year}"
        query = f"""
        SELECT DISTINCT date
        FROM {table}
        WHERE secid = {secid}
        """
        try:
            df = db.raw_sql(query)
            if not df.empty:
                surface_dates.extend(pd.to_datetime(df['date']))
        except Exception as e:
            print(f"Skipping table {table} due to error: {e}")

    if not surface_dates:
        return None

    sorted_dates = sorted(set(d for d in surface_dates if d < edate))

    if len(sorted_dates) < n_lag:
        return None

    return sorted_dates[-n_lag].strftime('%Y-%m-%d')

def get_earnings_data(ticker_list, start_date, end_date, db):
    """
    Fetches quarterly earnings announcement dates and related data
    for a list of tickers within a specific date range.

    Parameters:
    - ticker_list (list of str): List of stock tickers (e.g., ['AAPL', 'MSFT']).
    - start_date (str): Start date in 'YYYY-MM-DD' format.
    - end_date (str): End date in 'YYYY-MM-DD' format.
    - db: WRDS database connection object.

    Returns:
    - pandas.DataFrame: Table of earnings data with columns [cusip, ticker, datadate, earnings_date, fyearq, fqtr].
    """

    if not ticker_list:
        return pd.DataFrame(columns=["cusip", "ticker", "datadate", "earnings_date", "fyearq", "fqtr"])

    formatted_tickers = "', '".join(ticker_list)

    query_earning = f"""
    SELECT cusip,
           tic as ticker,
           datadate,
           rdq as earnings_date,
           fyearq,
           fqtr
    FROM comp.fundq
    WHERE tic IN ('{formatted_tickers}')
      AND rdq BETWEEN '{start_date}' AND '{end_date}'
      AND rdq IS NOT NULL
    """

    return db.raw_sql(query_earning)

def extract_term_diff_feature(secid, earnings_date, db, n_lag=20):
    """
    Computes the ratio of 30-day ATM implied volatility to 10-day ATM implied volatility.
    """
    query_date = get_relative_surface_date(secid, earnings_date, n_lag, db)
    if query_date is None:
        return None, None

    df = get_surface_row(secid, query_date, db)
    if df.empty:
        return None, query_date

    df_30 = df[df['days'] == 30]
    df_10 = df[df['days'] == 10]

    if df_30.empty or df_10.empty:
        return None, query_date

    call_30 = df_30[(df_30['cp_flag'] == 'C') & (df_30['delta'] == 50.0)]
    put_30 = df_30[(df_30['cp_flag'] == 'P') & (df_30['delta'] == -50.0)]
    call_10 = df_10[(df_10['cp_flag'] == 'C') & (df_10['delta'] == 50.0)]
    put_10 = df_10[(df_10['cp_flag'] == 'P') & (df_10['delta'] == -50.0)]

    if call_30.empty or put_30.empty or call_10.empty or put_10.empty:
        return None, query_date

    iv_30 = (call_30['impl_volatility'].mean() + put_30['impl_volatility'].mean()) / 2
    iv_10 = (call_10['impl_volatility'].mean() + put_10['impl_volatility'].mean()) / 2

    if iv_10 == 0:
        return None, query_date

    term_diff_ratio = iv_30 / iv_10
    return term_diff_ratio, query_date

def extract_smirk_feature(secid, earnings_date, db, n_lag=20):
    query_date = get_relative_surface_date(secid, earnings_date, n_lag, db)
    if query_date is None:
        return None, None

    df = get_surface_row(secid, query_date, db)
    df = df[df['days'] == 30]

    if df.empty:
        return None, query_date

    put_otm = df[(df['cp_flag'] == 'P') & (df['delta'] == -25.0)]
    call_atm = df[(df['cp_flag'] == 'C') & (df['delta'] == 50.0)]

    if put_otm.empty or call_atm.empty:
        return None, query_date

    iv_put = put_otm['impl_volatility'].mean()
    iv_call = call_atm['impl_volatility'].mean()

    # Normalized smirk = (put - call) / call
    smirk = (iv_put - iv_call) / iv_call if iv_call != 0 else None
    return smirk, query_date


