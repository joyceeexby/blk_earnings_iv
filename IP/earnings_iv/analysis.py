# analysis.py
import numpy as np
import pandas as pd
from scipy import stats
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import statsmodels.api as sm

def kernel_regression_iv(tte, ivs, target_tte, bandwidth=None):
    """
    Kernel regression to estimate normative IV at a target time-to-expiry (tte),
    following Wolfe's approach. Uses a Gaussian kernel.
    Args:
        tte: Array of time-to-expiry values (in days)
        ivs: Corresponding implied volatilities
        target_tte: The TTE at which to estimate the normative IV
        bandwidth: Kernel bandwidth (defaults to std of tte)
    Returns:
        Estimated IV at target_tte
    """
    tte = np.asarray(tte)
    ivs = np.asarray(ivs)
    if bandwidth is None:
        bandwidth = np.std(tte) if len(tte) > 1 else 1.0
    weights = np.exp(-0.5 * ((tte - target_tte) / bandwidth) ** 2)
    weights /= weights.sum() if weights.sum() > 0 else 1.0
    return np.sum(weights * ivs)

def estimate_normative_iv_curve(option_df, earnings_date, min_days=14, max_days=60):
    """
    Estimate the normative IV curve (not affected by earnings) using kernel regression.
    Only use options expiring at least min_days after earnings, up to max_days.
    Args:
        option_df: DataFrame of options (must have 'exdate', 'date', 'impl_volatility', 'tte')
        earnings_date: The earnings announcement date (datetime)
        min_days: Minimum days after earnings to use for regression
        max_days: Maximum days after earnings to use for regression
    Returns:
        (tte_list, normative_iv_list): Arrays of TTEs and estimated normative IVs
    """
    # Filter for options expiring after earnings
    after_earnings = option_df[option_df['exdate'] > earnings_date]
    after_earnings = after_earnings[(after_earnings['tte'] >= min_days) & (after_earnings['tte'] <= max_days)]
    tte = after_earnings['tte'].values
    ivs = after_earnings['impl_volatility'].values
    if len(tte) < 3:
        return None, None  # Not enough data
    # Estimate normative IV at each TTE in the range
    tte_grid = np.arange(min_days, max_days + 1)
    normative_iv = [kernel_regression_iv(tte, ivs, t) for t in tte_grid]
    return tte_grid, normative_iv

def calculate_eiv(option_df, earnings_date, current_date):
    """
    Calculate Earnings-Induced Volatility (EIV) as in Wolfe et al.
    Args:
        option_df: DataFrame of options (must have 'exdate', 'date', 'impl_volatility', 'tte')
        earnings_date: The earnings announcement date (datetime)
        current_date: The current date (datetime)
    Returns:
        EIV value (float) or None if not computable
    """
    # Find the option expiring immediately after earnings
    after_opts = option_df[option_df['exdate'] > earnings_date]
    if after_opts.empty:
        return None
    next_expiry = after_opts['exdate'].min()
    obs_row = after_opts[after_opts['exdate'] == next_expiry].iloc[0]
    t_after = (next_expiry - current_date).days
    iv_obs = obs_row['impl_volatility']
    # Estimate normative IV at t_after
    tte_grid, norm_iv_list = estimate_normative_iv_curve(option_df, earnings_date)
    if tte_grid is None or norm_iv_list is None:
        return None
    norm_iv_list = np.array(norm_iv_list)
    # Interpolate normative IV at t_after
    norm_iv = np.interp(t_after, tte_grid, norm_iv_list)
    # EIV formula (annualized): sqrt((T/252) * (IV_obs^2 - IV_norm^2))
    eiv = np.sqrt(max(0, (t_after / 252.0) * (iv_obs ** 2 - norm_iv ** 2)))
    return eiv

def calculate_option_metrics(options, stock_prices=None):
    """
    Calculate additional option metrics - ONLY WITH REAL DATA
    """
    df = options.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['exdate'] = pd.to_datetime(df['exdate'])
    df['tte'] = (df['exdate'] - df['date']).dt.days
    df['underlying_price'] = np.nan
    if stock_prices is not None and not stock_prices.empty:
        stock_df = stock_prices.copy()
        stock_df['date'] = pd.to_datetime(stock_df['date'])
        stock_df = stock_df.rename(columns={'close': 'underlying_price'})
        if 'underlying_price' in stock_df.columns:
            df = df.merge(stock_df[['date', 'secid', 'underlying_price']], on=['date', 'secid'], how='left', suffixes=('', '_stock'))
    if 'best_bid' in df.columns and 'best_offer' in df.columns:
        df['mid_price'] = (df['best_bid'] + df['best_offer']) / 2
    if 'mid_price' in df.columns:
        df['underlying_price'] = df['underlying_price'].fillna(df['mid_price'])
    df['moneyness'] = df['strike_price'] / 100000.0
    if 'best_bid' in df.columns and 'best_offer' in df.columns:
        df['bid_ask_spread'] = np.where(
            df['best_bid'] > 0,
            (df['best_offer'] - df['best_bid']) / df['best_bid'],
            np.nan
        )
        df['bid_ask_spread'] = df['bid_ask_spread'].clip(lower=0)
    if 'moneyness' in df.columns:
        df['log_moneyness'] = np.log(df['moneyness'].clip(lower=0.01))
    return df

def apply_data_filters(df, min_volume=10, max_bid_ask_spread=0.5, tte_range=(7, 60), moneyness_range=(0.8, 1.2)):
    """
    Apply data quality filters - ONLY filter on available real data
    """
    initial_count = len(df)
    if 'volume' in df.columns:
        df = df[df['volume'].notna() & (df['volume'] >= min_volume)]
    if 'bid_ask_spread' in df.columns:
        df = df[df['bid_ask_spread'].notna() & (df['bid_ask_spread'] <= max_bid_ask_spread)]
    if 'tte' in df.columns:
        df = df[(df['tte'] >= tte_range[0]) & (df['tte'] <= tte_range[1])]
    if 'moneyness' in df.columns:
        df = df[(df['moneyness'] >= moneyness_range[0]) & (df['moneyness'] <= moneyness_range[1])]
    real_data_filters = [
        ('vega', lambda x: x > 0),
        ('best_bid', lambda x: x > 0),
        ('best_offer', lambda x: x > 0),
        ('impl_volatility', lambda x: x > 0)
    ]
    for col_name, filter_func in real_data_filters:
        if col_name in df.columns:
            df = df[df[col_name].notna() & df[col_name].apply(filter_func)]
    return df

def merge_earnings_options(earnings_securities, options_filtered, event_window_days=30):
    """
    Merge earnings dates with option data using secid
    """
    earnings = earnings_securities.copy()
    options = options_filtered.copy()
    earnings['earnings_date'] = pd.to_datetime(earnings['earnings_date'])
    options['date'] = pd.to_datetime(options['date'])
    merged_data = []
    for _, earning in earnings.iterrows():
        secid = earning['secid']
        earnings_date = earning['earnings_date']
        secid_options = options[
            (options['secid'] == secid) &
            (options['date'] >= earnings_date - timedelta(days=event_window_days)) &
            (options['date'] <= earnings_date)
        ].copy()
        if len(secid_options) > 0:
            secid_options['earnings_date'] = earnings_date
            secid_options['days_to_earnings'] = (earnings_date - secid_options['date']).dt.days
            secid_options['ticker'] = earning['ticker']
            merged_data.append(secid_options)
    if merged_data:
        return pd.concat(merged_data, ignore_index=True)
    else:
        return None

def print_options_summary(df):
    print(f"Total option contracts: {len(df):,}")
    print(f"Unique securities: {df['secid'].nunique():,}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    if 'impl_volatility' in df.columns:
        print("Implied Volatility Stats:")
        print(df['impl_volatility'].describe())

def compute_realized_vol(prices, event_dates, window=5):
    """
    Compute realized volatility after each event date.
    prices: DataFrame with columns ['date', 'secid', 'close']
    event_dates: DataFrame with columns ['secid', 'earnings_date']
    window: Number of days after earnings to compute realized vol
    Returns: DataFrame with ['secid', 'earnings_date', 'realized_vol']
    """
    results = []
    prices['date'] = pd.to_datetime(prices['date'])
    for _, row in event_dates.iterrows():
        secid = row['secid']
        event_date = pd.to_datetime(row['earnings_date'])
        mask = (prices['secid'] == secid) & (prices['date'] > event_date) & (prices['date'] <= event_date + pd.Timedelta(days=window))
        window_prices = prices[mask].sort_values('date')
        if len(window_prices) > 1:
            returns = window_prices['close'].pct_change().dropna()
            realized_vol = returns.std() * (252 ** 0.5)  # annualized
            results.append({'secid': secid, 'earnings_date': event_date, 'realized_vol': realized_vol})
    return pd.DataFrame(results)

def run_iv_rv_regression(df, iv_col='implied_vol', rv_col='realized_vol'):
    """
    Run OLS regression of realized vol on implied vol.
    """
    X = df[iv_col]
    y = df[rv_col]
    T = len(y)
    X = sm.add_constant(X)
    maxlags = int(4 * (T / 100) ** (2 / 9))
    model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags':maxlags})
    print(model.summary())
    return model

# 1-day realized volatility (absolute return on earnings day)
def compute_1day_realized_vol(prices, event_dates):
    results = []
    prices['date'] = pd.to_datetime(prices['date'])
    for _, row in event_dates.iterrows():
        secid = row['secid']
        event_date = pd.to_datetime(row['earnings_date'])
        mask = (prices['secid'] == secid) & (prices['date'] == event_date)
        window_prices = prices[mask].sort_values('date')
        if len(window_prices) > 0:
            returns = window_prices['close'].pct_change().dropna()
            realized_vol = returns.abs().iloc[0] * (252 ** 0.5) if not returns.empty else None
            results.append({'secid': secid, 'earnings_date': event_date, 'realized_vol': realized_vol})
    return pd.DataFrame(results)

def extract_iv_for_events(options_df, earnings_securities, iv_type='ATM', days_before=1):
    """
    For each earnings event, extract the ATM IV from the expiry just after the event,
    measured days_before the earnings date.
    Returns a DataFrame with ['secid', 'earnings_date', 'implied_vol']
    """
    results = []
    options_df['date'] = pd.to_datetime(options_df['date'])
    options_df['exdate'] = pd.to_datetime(options_df['exdate'])
    for _, row in earnings_securities.iterrows():
        secid = row['secid']
        earnings_date = pd.to_datetime(row['earnings_date'])
        # Find the last available date before earnings
        obs_date = earnings_date - pd.Timedelta(days=days_before)
        opts = options_df[(options_df['secid'] == secid) & (options_df['date'] == obs_date)]
        if opts.empty:
            continue
        # Find the expiry just after earnings
        after_earnings = opts[opts['exdate'] > earnings_date]
        if after_earnings.empty:
            continue
        expiry = after_earnings['exdate'].min()
        atm_opts = after_earnings[after_earnings['exdate'] == expiry]
        # Use ATM IV (moneyness closest to 1)
        if 'moneyness' in atm_opts.columns:
            atm_row = atm_opts.iloc[(atm_opts['moneyness'] - 1).abs().argsort()[:1]]
        else:
            atm_row = atm_opts.iloc[:1]
        iv = atm_row['impl_volatility'].values[0]
        results.append({'secid': secid, 'earnings_date': earnings_date, 'implied_vol': iv})
    return pd.DataFrame(results)

def run_iv_predicts_rv_workflow(options_df, stock_prices, earnings_securities, window=5, days_before=1):
    """
    Extract ATM IV and realized vol for each earnings event, run regression, and print results.
    Keeps main.py clean!
    """
    iv_df = extract_iv_for_events(options_df, earnings_securities, days_before=days_before)
    rv_df = compute_realized_vol(stock_prices, earnings_securities, window=window)
    merged = iv_df.merge(rv_df, on=['secid', 'earnings_date'], how='inner')
    # Ensure numeric and drop NaNs
    merged['implied_vol'] = pd.to_numeric(merged['implied_vol'], errors='coerce')
    merged['realized_vol'] = pd.to_numeric(merged['realized_vol'], errors='coerce')
    merged = merged.dropna(subset=['implied_vol', 'realized_vol'])
    print(f"\nRegression of {window}-day realized vol on ATM IV (measured {days_before} day(s) before earnings):")
    if merged.empty:
        print("No data available for regression after filtering for numeric values.")
        return merged
    run_iv_rv_regression(merged, iv_col='implied_vol', rv_col='realized_vol')
    return merged  # Optionally return merged data for further analysis

# Add other analysis/statistics functions as needed 