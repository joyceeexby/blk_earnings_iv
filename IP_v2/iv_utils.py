import numpy as np
import pandas as pd
from datetime import timedelta

def fetch_iv_data(ticker, analysis_date, underlying_price=None, db_connection=None):
    """
    Fetch implied volatility data directly from WRDS for a given ticker and analysis date.
    Requires an open WRDS db_connection (mandatory).
    Returns a DataFrame with columns: tte, moneyness, put_iv, call_iv.
    """
    if db_connection is None:
        raise ValueError("A valid WRDS db_connection must be provided to fetch_iv_data.")
    try:
        db = db_connection

        # Get secid for the ticker
        secid_query = f"""
        SELECT DISTINCT secid
        FROM optionm.securd1
        WHERE ticker = '{ticker}'
          AND exchange_d != 0
        LIMIT 1
        """
        secid_result = db.raw_sql(secid_query)
        if isinstance(secid_result, pd.DataFrame):
            secid_df = secid_result
        else:
            secid_df = pd.DataFrame([dict(row) for row in secid_result])

        if secid_df.empty:
            print(f"Could not find secid for {ticker}")
            return None

        secid = secid_df.iloc[0]['secid']
        print(f"Found secid {secid} for {ticker}")

        # Get available tables first
        tables_query = f"""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'optionm'
          AND table_name LIKE 'opprcd%%'
        ORDER BY table_name
        """
        tables_result = db.raw_sql(tables_query)
        if isinstance(tables_result, pd.DataFrame):
            tables_df = tables_result
        else:
            tables_df = pd.DataFrame([dict(row) for row in tables_result])

        available_tables = set(tables_df['table_name'].str.lower())

        # Get options data around the analysis date - expand range
        analysis_date = pd.to_datetime(analysis_date)
        start_date = (analysis_date - timedelta(days=15)).strftime('%Y-%m-%d')
        end_date = (analysis_date + timedelta(days=15)).strftime('%Y-%m-%d')

        print(f"  Looking for IV data from {start_date} to {end_date}")

        # Build query using available tables
        year = analysis_date.year
        table_name = f"opprcd{year}"

        if table_name not in available_tables:
            # Try the base table name
            table_name = "opprcd"
            if table_name not in available_tables:
                print(f"Available tables: {sorted(available_tables)}")
                print(f"Could not find options table for year {year}")
                return None

        print(f"  Using table: {table_name}")

        # Simplified query - just get the essential IV data
        iv_query = f"""
        SELECT date, exdate, strike_price, cp_flag, impl_volatility
        FROM optionm.{table_name}
        WHERE secid = {secid}
          AND date BETWEEN '{start_date}' AND '{end_date}'
          AND impl_volatility > 0
          AND impl_volatility < 5.0  -- Filter out extreme values
        ORDER BY date, exdate, strike_price
        """

        print(f"  Executing IV query...")
        iv_result = db.raw_sql(iv_query)
        if isinstance(iv_result, pd.DataFrame):
            iv_df = iv_result
        else:
            iv_df = pd.DataFrame([dict(row) for row in iv_result])

        print(f"  Raw IV data: {len(iv_df)} records")

        if iv_df.empty:
            print(f"No IV data found for {ticker} on {analysis_date}")
            print(f"  Tried date range: {start_date} to {end_date}")
            print(f"  Used table: {table_name}")
            return None

        # Find the closest date to analysis_date
        iv_df['date'] = pd.to_datetime(iv_df['date'])
        iv_df['exdate'] = pd.to_datetime(iv_df['exdate'])

        print(f"  After date conversion: {len(iv_df)} records")

        # Get the closest date to analysis_date
        date_diff = abs(iv_df['date'] - analysis_date)
        closest_date = iv_df.loc[date_diff.idxmin(), 'date']

        print(f"  Closest date to {analysis_date.strftime('%Y-%m-%d')}: {closest_date.strftime('%Y-%m-%d')}")

        # Filter for the closest date
        iv_data = iv_df[iv_df['date'] == closest_date].copy()

        print(f"  After filtering to closest date: {len(iv_data)} records")

        if iv_data.empty:
            print(f"No IV data for {ticker} on {closest_date}")
            return None

        # Calculate moneyness and TTE
        if underlying_price is None:
            # Get stock price for the date
            stock_query = f"""
            SELECT close
            FROM optionm.secprd
            WHERE secid = {secid}
              AND date = '{closest_date}'
            """
            stock_result = db.raw_sql(stock_query)
            if isinstance(stock_result, pd.DataFrame):
                stock_df = stock_result
            else:
                stock_df = pd.DataFrame([dict(row) for row in stock_result])

            if not stock_df.empty:
                underlying_price = stock_df.iloc[0]['close']
            else:
                underlying_price = 100.0  # Default fallback

        print(f"  Using underlying price: ${underlying_price:.2f}")

        iv_data['underlying_price'] = underlying_price
        # Fix: Strike prices are in cents, so divide by 1000 to get dollars
        iv_data['moneyness'] = (iv_data['strike_price'] / 1000) / underlying_price
        iv_data['tte'] = (iv_data['exdate'] - iv_data['date']).dt.days

        print(f"  After calculating moneyness/TTE: {len(iv_data)} records")
        print(f"  Moneyness range: {iv_data['moneyness'].min():.3f} - {iv_data['moneyness'].max():.3f}")
        print(f"  TTE range: {iv_data['tte'].min():.0f} - {iv_data['tte'].max():.0f} days")

        # Filter for reasonable moneyness and TTE
        iv_data = iv_data[
            (iv_data['moneyness'].between(0.8, 1.2)) &
            (iv_data['tte'].between(10, 90))
        ]

        print(f"  After moneyness/TTE filtering: {len(iv_data)} records")

        # Separate puts and calls
        puts = iv_data[iv_data['cp_flag'] == 'P'].copy()
        calls = iv_data[iv_data['cp_flag'] == 'C'].copy()

        print(f"  Puts: {len(puts)} records, Calls: {len(calls)} records")

        # Create final data with put IV (more liquid for earnings)
        final_data = puts[['tte', 'moneyness', 'impl_volatility']].copy()
        final_data.columns = ['tte', 'moneyness', 'put_iv']

        # Add call IV if available
        if not calls.empty:
            call_data = calls[['tte', 'moneyness', 'impl_volatility']].copy()
            call_data.columns = ['tte', 'moneyness', 'call_iv']
            final_data = final_data.merge(call_data, on=['tte', 'moneyness'], how='left')
        else:
            final_data['call_iv'] = final_data['put_iv']  # Use put IV as fallback

        print(f"Fetched IV data with {len(final_data)} points")
        if len(final_data) > 0:
            print(f"TTE range: {final_data['tte'].min():.0f} - {final_data['tte'].max():.0f} days")
            print(f"Moneyness range: {final_data['moneyness'].min():.3f} - {final_data['moneyness'].max():.3f}")
        else:
            print("TTE range: nan - nan days")
            print("Moneyness range: <NA> - <NA>")

        return final_data

    except Exception as e:
        print(f"Error fetching IV data: {e}")
        return None

def find_volatility_kink_iv_ratio(iv_surface_data, normative_curve, earnings_date, analysis_date, kink_range=(20, 40)):
    """
    Find the maximum IV ratio (actual IV / normative IV) in the kink range around the earnings event.

    Args:
        iv_surface_data: DataFrame of IV surface data
        normative_curve: Tuple of (tte_list, normative_iv_list)
        earnings_date: Earnings announcement date (datetime or str)
        analysis_date: Analysis date (datetime or str)
        kink_range: Expected range for kink (days before earnings)

    Returns:
        The maximum IV ratio (float), or None if not found.
    """
    # Ensure dates are datetime
    earnings_date = pd.to_datetime(earnings_date)
    analysis_date = pd.to_datetime(analysis_date)

    tte_list, normative_iv_list = normative_curve

    # Find options in the kink range
    kink_options = iv_surface_data[
        (iv_surface_data['tte'] >= kink_range[0]) &
        (iv_surface_data['tte'] <= kink_range[1])
    ].copy()

    if kink_options.empty:
        print(f"No options found in kink range {kink_range}")
        return None

    # Calculate IV ratios (actual IV / normative IV)
    kink_options['iv_ratio'] = kink_options['put_iv'] / kink_options['tte'].apply(
        lambda x: np.interp(x, tte_list, normative_iv_list) if x <= max(tte_list) else normative_iv_list[-1]
    )

    # Return the maximum IV ratio (the kink)
    return kink_options['iv_ratio'].max() 