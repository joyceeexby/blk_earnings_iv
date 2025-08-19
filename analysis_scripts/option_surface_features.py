#!/usr/bin/env python3
"""
Option Surface Features Integration
Enhanced version of option surface features to be used in parallel with IEVR analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import wrds

class OptionSurfaceFeatures:
    """
    Calculate option surface features for earnings volatility analysis.
    Features include: TERM_RATIO, SKEW, KURT, IV_RATIO, SMIRK
    """
    
    def __init__(self, db_connection):
        self.db = db_connection
        
    def get_latest_ticker_info(self, ticker_list):
        """
        Fetches the latest effect_date, secid, and ticker from optionm_all.secnmd table
        for the given list of tickers, keeping only the latest secid per ticker.
        """
        if not ticker_list:
            return pd.DataFrame(columns=["ticker", "secid", "effect_date"])

        formatted_tickers = "', '".join(ticker_list)

        query_ticker = f"""
        SELECT ticker, secid, effect_date
        FROM optionm_all.secnmd
        WHERE ticker IN ('{formatted_tickers}')
        ORDER BY ticker, effect_date DESC;
        """

        df = self.db.raw_sql(query_ticker)

        # Keep only the row with the latest effect_date for each ticker
        df = df.sort_values(by=['ticker', 'effect_date'], ascending=[True, False])
        df = df.groupby('ticker').head(1).reset_index(drop=True)

        return df
    
    def get_surface_row(self, secid, query_date):
        """Get option surface data for a specific secid and date."""
        year = pd.to_datetime(query_date).year
        table_name = f"optionm_all.vsurfd{year}"

        query = f"""
        SELECT *
        FROM {table_name}
        WHERE secid = {secid}
          AND date = '{query_date}'
          AND days BETWEEN 7 AND 60
        """
        return self.db.raw_sql(query)
    
    def get_relative_surface_date(self, secid, earnings_date, n_lag):
        """
        Get the trading date that is `n_lag` trading days before the earnings date.
        Only uses available dates in optionm_all.vsurfd tables for this secid.
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
                df = self.db.raw_sql(query)
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
    
    def extract_skew_feature(self, secid, earnings_date, n_lag=15):
        """Extract skew feature from option surface."""
        query_date = self.get_relative_surface_date(secid, earnings_date, n_lag)
        if query_date is None:
            return None, None

        df = self.get_surface_row(secid, query_date)
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
    
    def extract_kurtosis_feature(self, secid, earnings_date, n_lag=15):
        """Extract kurtosis feature from option surface."""
        query_date = self.get_relative_surface_date(secid, earnings_date, n_lag)
        if query_date is None:
            return None, None

        df = self.get_surface_row(secid, query_date)
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
    
    def extract_term_diff_feature(self, secid, earnings_date, n_lag=15):
        """Extract term structure difference feature."""
        query_date = self.get_relative_surface_date(secid, earnings_date, n_lag)
        if query_date is None:
            return None, None

        df = self.get_surface_row(secid, query_date)
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
    
    def monthly_iv_change_ratio_feature(self, secid, earnings_date, n_lag=15, monthly_lag=21):
        """Extract monthly IV change ratio feature."""
        query_date = self.get_relative_surface_date(secid, earnings_date, n_lag)
        earlier_date = self.get_relative_surface_date(secid, earnings_date, n_lag + monthly_lag)

        if query_date is None or earlier_date is None:
            return None, query_date, earlier_date

        df_recent = self.get_surface_row(secid, query_date)
        df_earlier = self.get_surface_row(secid, earlier_date)

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
    
    def extract_smirk_feature(self, secid, earnings_date, n_lag=15):
        """Extract smirk feature from option surface."""
        query_date = self.get_relative_surface_date(secid, earnings_date, n_lag)
        if query_date is None:
            return None, None

        df = self.get_surface_row(secid, query_date)
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
    
    def calculate_surface_features(self, ticker, secid, earnings_date, n_lag=15):
        """
        Calculate all option surface features for a single earnings event.
        
        Returns:
            dict: Dictionary containing all calculated features
        """
        print(f"  Calculating option surface features for {ticker}...")
        
        features = {
            'ticker': ticker,
            'secid': secid,
            'earnings_date': earnings_date,
            'surface_date': None,
            'TERM_RATIO': None,
            'SKEW': None,
            'KURT': None,
            'IV_RATIO': None,
            'SMIRK': None
        }
        
        try:
            # Get surface date
            surface_date = self.get_relative_surface_date(secid, earnings_date, n_lag)
            features['surface_date'] = surface_date
            
            if surface_date is None:
                print(f"    ✗ No surface date available")
                return features
            
            # Calculate TERM_RATIO
            term_ratio, _ = self.extract_term_diff_feature(secid, earnings_date, n_lag)
            features['TERM_RATIO'] = term_ratio
            
            # Calculate SKEW
            skew, _ = self.extract_skew_feature(secid, earnings_date, n_lag)
            features['SKEW'] = skew
            
            # Calculate KURT
            kurt, _ = self.extract_kurtosis_feature(secid, earnings_date, n_lag)
            features['KURT'] = kurt
            
            # Calculate IV_RATIO
            iv_ratio, _, _ = self.monthly_iv_change_ratio_feature(secid, earnings_date, n_lag)
            features['IV_RATIO'] = iv_ratio
            
            # Calculate SMIRK
            smirk, _ = self.extract_smirk_feature(secid, earnings_date, n_lag)
            features['SMIRK'] = smirk
            
            print(f"    ✓ Surface features calculated successfully")
            print(f"      TERM_RATIO: {term_ratio:.4f}, SKEW: {skew:.4f}, KURT: {kurt:.4f}")
            print(f"      IV_RATIO: {iv_ratio:.4f}, SMIRK: {smirk:.4f}")
            
        except Exception as e:
            print(f"    ✗ Error calculating surface features: {e}")
        
        return features

