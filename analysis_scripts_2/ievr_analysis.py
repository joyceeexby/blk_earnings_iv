#!/usr/bin/env python3
"""
Implied Earnings Volatility Ratio (IEVR) Analysis
Step 2: Calculate IEVR for AAPL October 2022 earnings event using kernel regression
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy import stats

class IEVRAnalysis:
    """
    Calculate Implied Earnings Volatility Ratio (IEVR)
    IEVR = IV at kink / Normative IV at maturity
    """
    
    def __init__(self, db_connection=None):
        self.db = db_connection
        self.iv_surface = None
        self.earnings_date = None
        self.analysis_date = None
        self.underlying_price = None  # Store the fetched underlying price
        
    def kernel_regression_iv(self, tte, ivs, target_tte, bandwidth=None):
        """
        Kernel regression to estimate normative IV at a target time-to-expiry.
        Uses a Gaussian kernel.
        
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
        
        # Gaussian kernel weights
        weights = np.exp(-0.5 * ((tte - target_tte) / bandwidth) ** 2)
        weights /= weights.sum() if weights.sum() > 0 else 1.0
        
        return np.sum(weights * ivs)
    
    def estimate_normative_iv_curve(self, iv_surface_data, earnings_date, min_days=10, max_days=90):
        """
        Estimate the normative IV curve (not affected by earnings) using kernel regression.
        Only use options expiring before the earnings date.
        Args:
            iv_surface_data: DataFrame of IV surface data
            earnings_date: The earnings announcement date (datetime)
            min_days: Minimum days to expiry to use for regression
            max_days: Maximum days to expiry to use for regression
        Returns:
            (tte_list, normative_iv_list): Arrays of TTEs and estimated normative IVs
        """
        days_to_earnings = (earnings_date - self.analysis_date).days
        # Use only ATM options (tight filter)
        atm = iv_surface_data[iv_surface_data['moneyness'].between(0.98, 1.02)]
        before_earnings = atm[atm['tte'] < days_to_earnings]
        before_earnings = before_earnings[(before_earnings['tte'] >= min_days) & (before_earnings['tte'] <= max_days)]
        print("\n[DEBUG] Normative IV options sample:")
        debug_cols = [col for col in ['strike_price', 'underlying_price', 'moneyness', 'put_iv', 'tte', 'cp_flag'] if col in before_earnings.columns]
        print(before_earnings[debug_cols].head(10))
        if len(before_earnings) < 3:
            print(f"Not enough data for normative IV estimation: {len(before_earnings)} points")
            return None, None
        tte = before_earnings['tte'].values
        ivs = before_earnings['put_iv'].values
        tte_grid = np.arange(min_days, min(days_to_earnings, max_days) + 1, 5)  # 5-day intervals up to earnings
        normative_iv = [self.kernel_regression_iv(tte, ivs, t) for t in tte_grid]
        return tte_grid, normative_iv
    
    def find_volatility_kink(self, iv_surface_data, normative_curve, earnings_date, kink_range=(1, 20)):
        """
        Find the kink in the volatility surface around the earnings event.
        Args:
            iv_surface_data: DataFrame of IV surface data
            normative_curve: Tuple of (tte_list, normative_iv_list)
            earnings_date: Earnings announcement date
            kink_range: Expected range for kink (days after earnings)
        Returns:
            Dictionary with kink information
        """
        tte_list, normative_iv_list = normative_curve
        days_to_earnings = (earnings_date - self.analysis_date).days
        # Use only ATM options (tight filter)
        atm = iv_surface_data[iv_surface_data['moneyness'].between(0.98, 1.02)]
        kink_options = atm[(atm['tte'] > days_to_earnings + kink_range[0] - 1) & (atm['tte'] <= days_to_earnings + kink_range[1])].copy()
        print("\n[DEBUG] Kink IV options sample:")
        debug_cols = [col for col in ['strike_price', 'underlying_price', 'moneyness', 'put_iv', 'tte', 'cp_flag'] if col in kink_options.columns]
        print(kink_options[debug_cols].head(10))
        if kink_options.empty:
            print(f"No options found in kink range {kink_range} days after earnings")
            return None
        kink_options['iv_ratio'] = kink_options['put_iv'] / kink_options['tte'].apply(
            lambda x: np.interp(x, tte_list, normative_iv_list) if x <= max(tte_list) else normative_iv_list[-1]
        )
        max_ratio_idx = kink_options['iv_ratio'].idxmax()
        kink_point = kink_options.loc[max_ratio_idx]
        kink_tte = kink_point['tte']
        normative_iv_at_kink = np.interp(kink_tte, tte_list, normative_iv_list)
        return {
            'tte': kink_tte,
            'actual_iv': kink_point['put_iv'],
            'normative_iv': normative_iv_at_kink,
            'iv_ratio': kink_point['iv_ratio'],
            'moneyness': kink_point['moneyness'],
            'days_to_earnings': days_to_earnings - kink_tte
        }
    
    def calculate_ievr(self, ticker, earnings_date, analysis_days_before=30, underlying_price=None, include_spx=False):
        """
        Calculate Implied Earnings Volatility Ratio (IEVR).
        Args:
            ticker: Stock ticker
            earnings_date: Earnings announcement date
            analysis_days_before: Days before earnings to analyze (default 30)
            underlying_price: Stock price at analysis date (if None, fetch automatically)
            include_spx: Whether to also calculate S&P 500 IEVR (default False - disabled to save time)
        Returns:
            Dictionary with IEVR and analysis details
        """
        print(f"\n{'='*80}")
        print(f"IEVR ANALYSIS: {ticker} - {earnings_date}")
        print(f"{'='*80}")
        earnings_date = pd.to_datetime(earnings_date)
        self.earnings_date = earnings_date
        self.analysis_date = earnings_date - timedelta(days=analysis_days_before)
        print(f"Analysis date: {self.analysis_date.strftime('%Y-%m-%d')} ({analysis_days_before} days before earnings)")
        print(f"Earnings date: {earnings_date.strftime('%Y-%m-%d')}")
        # Fetch IV surface and underlying price
        surface = DirectIVData(ticker, self.analysis_date.strftime('%Y-%m-%d'), None, self.db)
        iv_surface_data = surface.fetch_iv_data()
        if iv_surface_data is None or iv_surface_data.empty:
            print(f"No IV data found for {ticker} on {self.analysis_date}")
            return None
        if hasattr(surface, 'underlying_price') and surface.underlying_price is not None:
            self.underlying_price = surface.underlying_price
        elif 'underlying_price' in iv_surface_data.columns:
            self.underlying_price = iv_surface_data['underlying_price'].iloc[0]
        else:
            print("Warning: Could not fetch underlying price, using fallback 100.0")
            self.underlying_price = 100.0
        print(f"[DEBUG] Underlying price used: {self.underlying_price}")
        if iv_surface_data is None:
            print("Failed to create IV surface")
            return None
        self.iv_surface = iv_surface_data
        print("[DEBUG] Moneyness distribution (all options):")
        print(iv_surface_data['moneyness'].describe())
        # ATM options only
        atm = iv_surface_data[iv_surface_data['moneyness'].between(0.98, 1.02)]
        days_to_earnings = (earnings_date - self.analysis_date).days
        # Pre-earnings: TTE < days_to_earnings
        pre_mask = (atm['tte'] < days_to_earnings)
        pre_earnings = atm[pre_mask]
        # Post-earnings: TTE in (days_to_earnings, days_to_earnings+20]
        post_mask = (atm['tte'] > days_to_earnings) & (atm['tte'] <= days_to_earnings + 20)
        post_earnings = atm[post_mask]
        # Kernel regression for each window
        def kernel_avg_iv(df):
            if len(df) < 2:
                return float('nan')
            tte = df['tte'].values
            ivs = df['put_iv'].values
            # Use a grid over the window
            tte_grid = np.linspace(df['tte'].min(), df['tte'].max(), 10)
            kr_vals = [self.kernel_regression_iv(tte, ivs, t) for t in tte_grid]
            return np.mean(kr_vals)
        avg_pre = kernel_avg_iv(pre_earnings)
        avg_post = kernel_avg_iv(post_earnings)
        print(f"[DEBUG] Pre-earnings kernel average IV: {avg_pre:.4f}")
        print(f"[DEBUG] Post-earnings kernel average IV: {avg_post:.4f}")
        if np.isnan(avg_pre) or np.isnan(avg_post) or avg_pre == 0:
            print("Not enough data for IEVR calculation.")
            return None
        ievr = avg_post / avg_pre
        # Calculate skew ratio (95Put IV / 105Call IV)
        skew_ratio = self.calculate_skew_ratio(iv_surface_data, self.underlying_price)
        
        # Calculate S&P 500 IEVR if requested
        spx_ievr = None
        if include_spx:
            spx_ievr = self.calculate_spx_ievr(earnings_date, analysis_days_before)
        
        results = {
            'ticker': ticker,
            'earnings_date': earnings_date,
            'analysis_date': self.analysis_date,
            'ievr': ievr,
            'avg_pre': avg_pre,
            'avg_post': avg_post,
            'normative_implied_vol': avg_pre,  # Added normative implied vol (pre-earnings avg)
            'skew_ratio': skew_ratio,  # Added skew ratio
            'spx_ievr': spx_ievr,  # Added S&P 500 IEVR
            'iv_surface_data': iv_surface_data,
            'days_to_earnings': days_to_earnings
        }
        # self.plot_ievr_analysis(results) # Disabled for batch mode
        print(f"\nIEVR Analysis Summary:")
        print(f"  IEVR: {ievr:.3f}")
        print(f"  Pre-earnings kernel avg IV: {avg_pre:.3f}")
        print(f"  Post-earnings kernel avg IV: {avg_post:.3f}")
        print(f"  Days to Earnings: {days_to_earnings}")
        print(f"  Skew Ratio (90Put/110Call): {skew_ratio:.3f}")
        return results
    
    def calculate_skew_ratio(self, iv_surface_data, underlying_price):
        """
        Calculate skew ratio: 95Put IV / 105Call IV.
        
        Args:
            iv_surface_data: DataFrame with IV surface data
            underlying_price: Current stock price
            
        Returns:
            Skew ratio (float)
        """
        try:
            # Calculate moneyness for 90Put and 110Call
            put_90_moneyness = 0.90  # 90% of underlying price
            call_110_moneyness = 1.10  # 110% of underlying price
            
            # Find 90Put (closest to 0.90 moneyness)
            if 'put_iv' in iv_surface_data.columns:
                # Use put_iv column directly
                puts = iv_surface_data[iv_surface_data['put_iv'].notna()].copy()
                print(f"  Debug: Found {len(puts)} put options")
                if len(puts) > 0:
                    print(f"  Debug: Put moneyness range: {puts['moneyness'].min():.3f} - {puts['moneyness'].max():.3f}")
                    puts['moneyness_diff'] = abs(puts['moneyness'] - put_90_moneyness)
                    put_90_data = puts.loc[puts['moneyness_diff'].idxmin()]
                    put_90_iv = put_90_data['put_iv']
                    put_90_actual_moneyness = put_90_data['moneyness']
                    print(f"  Debug: Selected put with moneyness {put_90_actual_moneyness:.3f} (target: {put_90_moneyness:.3f})")
                else:
                    print("Warning: No put options found for skew ratio calculation")
                    return np.nan
            else:
                print("Warning: put_iv column not found in IV surface data")
                return np.nan
            
            # Find 110Call (closest to 1.10 moneyness)
            if 'call_iv' in iv_surface_data.columns:
                # Use call_iv column directly
                calls = iv_surface_data[iv_surface_data['call_iv'].notna()].copy()
                print(f"  Debug: Found {len(calls)} call options")
                if len(calls) > 0:
                    print(f"  Debug: Call moneyness range: {calls['moneyness'].min():.3f} - {calls['moneyness'].max():.3f}")
                    calls['moneyness_diff'] = abs(calls['moneyness'] - call_110_moneyness)
                    call_110_data = calls.loc[calls['moneyness_diff'].idxmin()]
                    call_110_iv = call_110_data['call_iv']
                    call_110_actual_moneyness = call_110_data['moneyness']
                    print(f"  Debug: Selected call with moneyness {call_110_actual_moneyness:.3f} (target: {call_110_moneyness:.3f})")
                else:
                    print("Warning: No call options found for skew ratio calculation")
                    return np.nan
            else:
                print("Warning: call_iv column not found in IV surface data")
                return np.nan
            
            # Calculate skew ratio
            if call_110_iv > 0:
                skew_ratio = put_90_iv / call_110_iv
                print(f"  Skew calculation: {put_90_iv:.4f} / {call_110_iv:.4f} = {skew_ratio:.3f}")
                print(f"  Put moneyness: {put_90_actual_moneyness:.3f}, Call moneyness: {call_110_actual_moneyness:.3f}")
            else:
                print("Warning: Call IV is zero or negative")
                return np.nan
            
            return skew_ratio
            
        except Exception as e:
            print(f"Error calculating skew ratio: {e}")
            return np.nan
    
    def calculate_spx_ievr(self, earnings_date, analysis_days_before=30):
        """
        Calculate S&P 500 IEVR for the same earnings date.
        This provides a market-level volatility expectation for comparison.
        
        Args:
            earnings_date: Earnings announcement date
            analysis_days_before: Days before earnings to analyze
            
        Returns:
            S&P 500 IEVR value (float) or None if calculation fails
        """
        try:
            print(f"\nCalculating S&P 500 IEVR for {earnings_date.strftime('%Y-%m-%d')}...")
            
            # Use the same analysis date as the stock
            analysis_date = earnings_date - timedelta(days=analysis_days_before)
            
            # Fetch S&P 500 IV surface data
            spx_surface = DirectIVData('SPX', analysis_date.strftime('%Y-%m-%d'), None, self.db)
            spx_iv_surface_data = spx_surface.fetch_iv_data()
            
            if spx_iv_surface_data is None or spx_iv_surface_data.empty:
                print("No S&P 500 IV data found")
                return None
            
            # Get S&P 500 underlying price
            if hasattr(spx_surface, 'underlying_price') and spx_surface.underlying_price is not None:
                spx_underlying_price = spx_surface.underlying_price
            elif 'underlying_price' in spx_iv_surface_data.columns:
                spx_underlying_price = spx_iv_surface_data['underlying_price'].iloc[0]
            else:
                print("Warning: Could not fetch S&P 500 underlying price")
                return None
            
            # ATM options only for S&P 500
            atm_spx = spx_iv_surface_data[spx_iv_surface_data['moneyness'].between(0.98, 1.02)]
            days_to_earnings = (earnings_date - analysis_date).days
            
            # Pre-earnings: TTE < days_to_earnings
            pre_mask = (atm_spx['tte'] < days_to_earnings)
            pre_earnings_spx = atm_spx[pre_mask]
            
            # Post-earnings: TTE in (days_to_earnings, days_to_earnings+20]
            post_mask = (atm_spx['tte'] > days_to_earnings) & (atm_spx['tte'] <= days_to_earnings + 20)
            post_earnings_spx = atm_spx[post_mask]
            
            # Kernel regression for each window
            def kernel_avg_iv(df):
                if len(df) < 2:
                    return float('nan')
                tte = df['tte'].values
                ivs = df['put_iv'].values
                # Use a grid over the window
                tte_grid = np.linspace(df['tte'].min(), df['tte'].max(), 10)
                kr_vals = [self.kernel_regression_iv(tte, ivs, t) for t in tte_grid]
                return np.mean(kr_vals)
            
            avg_pre_spx = kernel_avg_iv(pre_earnings_spx)
            avg_post_spx = kernel_avg_iv(post_earnings_spx)
            
            if np.isnan(avg_pre_spx) or np.isnan(avg_post_spx) or avg_pre_spx == 0:
                print("Not enough S&P 500 data for IEVR calculation.")
                return None
            
            spx_ievr = avg_post_spx / avg_pre_spx
            
            print(f"S&P 500 IEVR: {spx_ievr:.3f}")
            print(f"  S&P 500 Pre-earnings avg IV: {avg_pre_spx:.3f}")
            print(f"  S&P 500 Post-earnings avg IV: {avg_post_spx:.3f}")
            
            return spx_ievr
            
        except Exception as e:
            print(f"Error calculating S&P 500 IEVR: {e}")
            return None
    
    def _print_ievr_analysis(self, results):
        """
        Print detailed IEVR analysis.
        """
        kink = results['kink_info']
        
        print(f"\nIEVR Analysis Results:")
        print(f"  Ticker: {results['ticker']}")
        print(f"  Analysis Date: {results['analysis_date'].strftime('%Y-%m-%d')}")
        print(f"  Earnings Date: {results['earnings_date'].strftime('%Y-%m-%d')}")
        print(f"  Days to Earnings: {kink['days_to_earnings']}")
        print(f"  Kink TTE: {kink['tte']:.0f} days")
        print(f"  Kink Moneyness: {kink['moneyness']:.3f}")
        print(f"  Actual IV at Kink: {kink['actual_iv']:.3f} ({kink['actual_iv']*100:.1f}%)")
        print(f"  Normative IV at Kink: {kink['normative_iv']:.3f} ({kink['normative_iv']*100:.1f}%)")
        print(f"  IEVR: {results['ievr']:.3f}")
        
        # Validate IEVR
        if 0.9 <= results['ievr'] <= 2.0:
            print(f"  ✓ IEVR is in expected range (0.9-2.0)")
        else:
            print(f"  ⚠ IEVR is outside expected range (0.9-2.0)")
        
        # Validate kink location
        if 20 <= kink['days_to_earnings'] <= 40:
            print(f"  ✓ Kink location is in expected range (20-40 days before earnings)")
        else:
            print(f"  ⚠ Kink location ({kink['days_to_earnings']:.0f} days) outside expected range")
    
    def plot_ievr_analysis(self, results):
        """
        Plot IEVR analysis with normative and post-earnings kernel averages and event marker.
        """
        if results is None:
            print("No data available for plotting")
            return
        iv_surface_data = results['iv_surface_data']
        days_to_earnings = results['days_to_earnings']
        import matplotlib.pyplot as plt
        fig, ax1 = plt.subplots(1, 1, figsize=(9, 6))
        atm_data = iv_surface_data[iv_surface_data['moneyness'].between(0.98, 1.02)]
        ax1.scatter(atm_data['tte'], atm_data['put_iv'] * 100, alpha=0.6, color='blue', s=20, label='Actual IV (ATM)')
        # Highlight pre-earnings and post-earnings windows
        pre = atm_data[atm_data['tte'] < days_to_earnings]
        post = atm_data[(atm_data['tte'] > days_to_earnings) & (atm_data['tte'] <= days_to_earnings + 20)]
        ax1.scatter(pre['tte'], pre['put_iv'] * 100, color='green', s=40, label='Pre-Earnings (used)')
        ax1.scatter(post['tte'], post['put_iv'] * 100, color='red', s=40, label='Post-Earnings (used)')
        # Mark kernel averages
        ax1.axhline(results['avg_pre'] * 100, color='green', linestyle='--', label='Pre-Earnings Kernel Avg')
        ax1.axhline(results['avg_post'] * 100, color='red', linestyle='--', label='Post-Earnings Kernel Avg')
        # Indicate earnings event
        ax1.axvline(days_to_earnings, color='black', linestyle='--', linewidth=2, label='Earnings Event')
        ax1.annotate('Earnings Event', xy=(days_to_earnings, ax1.get_ylim()[1]*0.95),
                     xytext=(days_to_earnings+2, ax1.get_ylim()[1]*0.95),
                     arrowprops=dict(arrowstyle='->', color='black'),
                     fontsize=10, color='black', ha='left')
        ax1.set_xlabel('Time to Expiration (Days)')
        ax1.set_ylabel('Implied Volatility (%)')
        ax1.set_title(f'{results["ticker"]} IV Surface: Pre/Post-Earnings Kernel Avg')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        plt.tight_layout()
        # plt.show()  # Disabled for batch mode
        print(f"\nIEVR Analysis Summary:")
        print(f"  IEVR: {results['ievr']:.3f}")
        print(f"  Pre-earnings kernel avg IV: {results['avg_pre']:.3f}")
        print(f"  Post-earnings kernel avg IV: {results['avg_post']:.3f}")
        print(f"  Days to Earnings: {results['days_to_earnings']:.0f}")
        if results.get('spx_ievr') is not None:
            print(f"  S&P 500 IEVR: {results['spx_ievr']:.3f}")
        else:
            print(f"  S&P 500 IEVR: Not calculated")

class DirectIVData:
    """
    Direct IV data fetcher for IEVR analysis - simpler than VolatilitySurface.
    """
    def __init__(self, ticker, analysis_date, underlying_price=None, db_connection=None):
        self.ticker = ticker
        self.analysis_date = pd.to_datetime(analysis_date)
        self.underlying_price = underlying_price
        self.db = db_connection
        self.iv_data = None
        
    def fetch_iv_data(self):
        """
        Fetch implied volatility data directly from WRDS.
        """
        try:
            # Use passed database connection or create new one
            if self.db is None:
                import wrds
                db = wrds.Connection(wrds_username="sami_sellami",
                                   password="xampok-9Hezfy-cahveq")
            else:
                db = self.db
            
            # Get secid for the ticker
            secid_query = f"""
            SELECT DISTINCT secid
            FROM optionm.securd1
            WHERE ticker = '{self.ticker}'
              AND exchange_d != 0
            LIMIT 1
            """
            secid_result = db.raw_sql(secid_query)
            if isinstance(secid_result, pd.DataFrame):
                secid_df = secid_result
            else:
                secid_df = pd.DataFrame([dict(row) for row in secid_result])
            
            if secid_df.empty:
                print(f"Could not find secid for {self.ticker}")
                if self.db is None:
                    db.close()
                return None
            
            secid = secid_df.iloc[0]['secid']
            print(f"Found secid {secid} for {self.ticker}")
            
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
            start_date = (self.analysis_date - timedelta(days=15)).strftime('%Y-%m-%d')
            end_date = (self.analysis_date + timedelta(days=15)).strftime('%Y-%m-%d')
            
            print(f"  Looking for IV data from {start_date} to {end_date}")
            
            # Build query using available tables
            year = self.analysis_date.year
            table_name = f"opprcd{year}"
            
            if table_name not in available_tables:
                # Try the base table name
                table_name = "opprcd"
                if table_name not in available_tables:
                    print(f"Available tables: {sorted(available_tables)}")
                    print(f"Could not find options table for year {year}")
                    if self.db is None:
                        db.close()
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
            
            if self.db is None:
                db.close()
            
            if iv_df.empty:
                print(f"No IV data found for {self.ticker} on {self.analysis_date}")
                print(f"  Tried date range: {start_date} to {end_date}")
                print(f"  Used table: {table_name}")
                return None
            
            # Find the closest date to analysis_date
            iv_df['date'] = pd.to_datetime(iv_df['date'])
            iv_df['exdate'] = pd.to_datetime(iv_df['exdate'])
            
            print(f"  After date conversion: {len(iv_df)} records")
            
            # Get the closest date to analysis_date
            date_diff = abs(iv_df['date'] - self.analysis_date)
            closest_date = iv_df.loc[date_diff.idxmin(), 'date']
            
            print(f"  Closest date to {self.analysis_date.strftime('%Y-%m-%d')}: {closest_date.strftime('%Y-%m-%d')}")
            
            # Filter for the closest date
            iv_data = iv_df[iv_df['date'] == closest_date].copy()
            
            print(f"  After filtering to closest date: {len(iv_data)} records")
            
            if iv_data.empty:
                print(f"No IV data for {self.ticker} on {closest_date}")
                return None
            
            # Calculate moneyness and TTE
            if self.underlying_price is None:
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
                    self.underlying_price = stock_df.iloc[0]['close']
                else:
                    self.underlying_price = 100.0  # Default fallback
            
            print(f"  Using underlying price: ${self.underlying_price:.2f}")
            
            iv_data['underlying_price'] = self.underlying_price
            # Fix: Strike prices are in cents, so divide by 1000 to get dollars
            iv_data['moneyness'] = (iv_data['strike_price'] / 1000) / self.underlying_price
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
            # Keep strike_price and underlying_price for debugging
            final_data = puts[['tte', 'moneyness', 'impl_volatility', 'strike_price', 'underlying_price', 'cp_flag']].copy()
            final_data.columns = ['tte', 'moneyness', 'put_iv', 'strike_price', 'underlying_price', 'cp_flag']
            
            # Add call IV if available
            if not calls.empty:
                call_data = calls[['tte', 'moneyness', 'impl_volatility', 'strike_price', 'underlying_price', 'cp_flag']].copy()
                call_data.columns = ['tte', 'moneyness', 'call_iv', 'strike_price', 'underlying_price', 'cp_flag']
                # Merge on tte and moneyness only (not strike_price since puts/calls have different strikes)
                final_data = final_data.merge(call_data[['tte', 'moneyness', 'call_iv']], on=['tte', 'moneyness'], how='left')
            else:
                final_data['call_iv'] = final_data['put_iv']  # Use put IV as fallback
            
            print(f"Fetched IV data with {len(final_data)} points")
            if len(final_data) > 0:
                print(f"TTE range: {final_data['tte'].min():.0f} - {final_data['tte'].max():.0f} days")
                print(f"Moneyness range: {final_data['moneyness'].min():.3f} - {final_data['moneyness'].max():.3f}")
            else:
                print("TTE range: nan - nan days")
                print("Moneyness range: <NA> - <NA>")
            
            self.iv_data = final_data
            return final_data
            
        except Exception as e:
            print(f"Error fetching IV data: {e}")
            return None

    def get_slice(self, tte=None, moneyness=None, tol=0.05):
        if self.iv_data is None:
            return pd.DataFrame()
        df = self.iv_data
        if tte is not None:
            df = df[np.abs(df['tte'] - tte) <= tol * tte]
        if moneyness is not None:
            df = df[np.abs(df['moneyness'] - moneyness) <= tol]
        return df

    def get_iv(self, tte, moneyness, tol=0.05):
        df = self.get_slice(tte=tte, moneyness=moneyness, tol=tol)
        if not df.empty:
            return df['put_iv'].mean()
        return np.nan

    def get_surface_grid(self):
        if self.iv_data is None:
            return pd.DataFrame()
        return self.iv_data[['tte', 'moneyness', 'put_iv']]

def main():
    """
    Main function to run IEVR analysis for AAPL October 2022.
    """
    print("IMPLIED EARNINGS VOLATILITY RATIO (IEVR) ANALYSIS")
    print("="*80)
    
    try:
        # Connect to WRDS
        import wrds
        db = wrds.Connection(wrds_username="sami_sellami", password="xampok-9Hezfy-cahveq")
        print("✓ Connected to WRDS")
        
        # Initialize analysis with database connection
        analyzer = IEVRAnalysis(db)
    
        # Calculate IEVR for AAPL October 2022
        results = analyzer.calculate_ievr(
            ticker='AAPL',
            earnings_date='2022-10-27',  # AAPL Q4 2022 earnings
            analysis_days_before=30,     # 30 days before earnings
            underlying_price=160.0       # Approximate price in September 2022
        )
        
        if results is not None:
            # Plot the IEVR analysis
            analyzer.plot_ievr_analysis(results)
            
            print(f"\n✓ IEVR analysis completed successfully!")
            print(f"  IEVR = {results['ievr']:.3f}")
            
            # Compare with REVR from Step 1
            revr = 1.424  # From Step 1
            print(f"\nComparison with REVR:")
            print(f"  REVR (realized): {revr:.3f}")
            print(f"  IEVR (implied): {results['ievr']:.3f}")
            print(f"  Ratio (IEVR/REVR): {results['ievr']/revr:.3f}")
            
        else:
            print("✗ IEVR analysis failed")
            
        # Close connection
        db.close()
        print("\n✓ Database connection closed")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 