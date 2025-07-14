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
        Only use options expiring at least min_days after earnings, up to max_days.
        
        Args:
            iv_surface_data: DataFrame of IV surface data
            earnings_date: The earnings announcement date (datetime)
            min_days: Minimum days after earnings to use for regression
            max_days: Maximum days after earnings to use for regression
        
        Returns:
            (tte_list, normative_iv_list): Arrays of TTEs and estimated normative IVs
        """
        # Filter for options expiring after earnings
        after_earnings = iv_surface_data[iv_surface_data['tte'] > (earnings_date - self.analysis_date).days]
        after_earnings = after_earnings[(after_earnings['tte'] >= min_days) & (after_earnings['tte'] <= max_days)]
        
        if len(after_earnings) < 3:
            print(f"Not enough data for normative IV estimation: {len(after_earnings)} points")
            return None, None
        
        # Use put IV for normative curve (more liquid)
        tte = after_earnings['tte'].values
        ivs = after_earnings['put_iv'].values
        
        # Estimate normative IV at each TTE in the range
        tte_grid = np.arange(min_days, max_days + 1, 5)  # 5-day intervals
        normative_iv = [self.kernel_regression_iv(tte, ivs, t) for t in tte_grid]
        
        return tte_grid, normative_iv
    
    def find_volatility_kink(self, iv_surface_data, normative_curve, earnings_date, kink_range=(20, 40)):
        """
        Find the kink in the volatility surface around the earnings event.
        
        Args:
            iv_surface_data: DataFrame of IV surface data
            normative_curve: Tuple of (tte_list, normative_iv_list)
            earnings_date: Earnings announcement date
            kink_range: Expected range for kink (days before earnings)
        
        Returns:
            Dictionary with kink information
        """
        tte_list, normative_iv_list = normative_curve
        
        # Calculate days to earnings for each TTE
        days_to_earnings = (earnings_date - self.analysis_date).days
        
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
        
        # Find the maximum IV ratio (the kink)
        max_ratio_idx = kink_options['iv_ratio'].idxmax()
        kink_point = kink_options.loc[max_ratio_idx]
        
        # Get normative IV at kink maturity
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
    
    def calculate_ievr(self, ticker, earnings_date, analysis_days_before=30, underlying_price=160.0):
        """
        Calculate Implied Earnings Volatility Ratio (IEVR).
        
        Args:
            ticker: Stock ticker
            earnings_date: Earnings announcement date
            analysis_days_before: Days before earnings to analyze
            underlying_price: Stock price at analysis date
        
        Returns:
            Dictionary with IEVR and analysis details
        """
        print(f"\n{'='*80}")
        print(f"IEVR ANALYSIS: {ticker} - {earnings_date}")
        print(f"{'='*80}")
        
        # Convert dates
        earnings_date = pd.to_datetime(earnings_date)
        self.earnings_date = earnings_date
        self.analysis_date = earnings_date - timedelta(days=analysis_days_before)
        
        print(f"Analysis date: {self.analysis_date.strftime('%Y-%m-%d')} ({analysis_days_before} days before earnings)")
        print(f"Earnings date: {earnings_date.strftime('%Y-%m-%d')}")
        
        # Create IV surface using shared connection
        print(f"\nCreating IV surface for analysis...")
        surface = DirectIVData(ticker, self.analysis_date.strftime('%Y-%m-%d'), underlying_price, self.db)
        iv_surface_data = surface.fetch_iv_data()
        
        if iv_surface_data is None:
            print("Failed to create IV surface")
            return None
        
        self.iv_surface = iv_surface_data
        
        # Estimate normative IV curve
        print(f"\nEstimating normative IV curve...")
        normative_curve = self.estimate_normative_iv_curve(iv_surface_data, earnings_date)
        
        if normative_curve[0] is None:
            print("Failed to estimate normative IV curve")
            return None
        
        tte_list, normative_iv_list = normative_curve
        print(f"Normative IV curve estimated for {len(tte_list)} TTE points")
        
        # Find volatility kink
        print(f"\nFinding volatility kink...")
        kink_info = self.find_volatility_kink(iv_surface_data, normative_curve, earnings_date)
        
        if kink_info is None:
            print("Failed to find volatility kink")
            return None
        
        # Calculate IEVR
        ievr = kink_info['iv_ratio']
        
        # Analysis results
        results = {
            'ticker': ticker,
            'earnings_date': earnings_date,
            'analysis_date': self.analysis_date,
            'kink_info': kink_info,
            'ievr': ievr,
            'normative_curve': normative_curve,
            'iv_surface_data': iv_surface_data
        }
        
        # Print analysis
        self._print_ievr_analysis(results)
        
        return results
    
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
        Plot IEVR analysis with normative curve and kink.
        """
        if results is None:
            print("No data available for plotting")
            return
        
        iv_surface_data = results['iv_surface_data']
        tte_list, normative_iv_list = results['normative_curve']
        kink = results['kink_info']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: IV Surface with Normative Curve
        # Plot actual IV surface (ATM options)
        atm_data = iv_surface_data[iv_surface_data['moneyness'].between(0.98, 1.02)]
        if not atm_data.empty:
            ax1.scatter(atm_data['tte'], atm_data['put_iv'] * 100, 
                       alpha=0.6, color='blue', s=20, label='Actual IV (ATM)')
        
        # Plot normative curve
        ax1.plot(tte_list, np.array(normative_iv_list) * 100, 'r-', 
                linewidth=2, label='Normative IV Curve')
        
        # Mark the kink
        ax1.scatter(kink['tte'], kink['actual_iv'] * 100, 
                   color='red', s=100, marker='*', label=f'Kink (IEVR={results["ievr"]:.3f})')
        
        ax1.set_xlabel('Time to Expiration (Days)')
        ax1.set_ylabel('Implied Volatility (%)')
        ax1.set_title(f'{results["ticker"]} IV Surface with Normative Curve')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: IV Ratio Analysis
        # Calculate IV ratios for all ATM options
        atm_data['iv_ratio'] = atm_data['put_iv'] / atm_data['tte'].apply(
            lambda x: np.interp(x, tte_list, normative_iv_list) if x <= max(tte_list) else normative_iv_list[-1]
        )
        
        ax2.scatter(atm_data['tte'], atm_data['iv_ratio'], 
                   alpha=0.6, color='green', s=20, label='IV Ratio')
        
        # Mark the kink
        ax2.scatter(kink['tte'], kink['iv_ratio'], 
                   color='red', s=100, marker='*', label=f'Kink (IEVR={results["ievr"]:.3f})')
        
        ax2.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Normative Level')
        ax2.set_xlabel('Time to Expiration (Days)')
        ax2.set_ylabel('IV Ratio (Actual / Normative)')
        ax2.set_title(f'{results["ticker"]} IV Ratio Analysis')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        print(f"\nIEVR Analysis Summary:")
        print(f"  IEVR: {results['ievr']:.3f}")
        print(f"  Kink TTE: {kink['tte']:.0f} days")
        print(f"  Days to Earnings: {kink['days_to_earnings']:.0f}")

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